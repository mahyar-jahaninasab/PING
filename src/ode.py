import gc
import torch
import os 
from model import MLP
from dotenv import load_dotenv
import json
import re
import ast
import torch.nn as nn
from filtering import FilterBestSpace

load_dotenv()
problem_path = os.getenv('PROBLEM')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(problem_path) as f:
    data = json.load(f)
DOMAIN      = data["DOMAIN"]
GENERATOR   = data["GENERATOR"]
REWARD      = data["REWARD"]
CONSTRAINTS = data["CONSTRAINTS"]

class odeFeasibility:

    def __init__(self, ode_requirements):
        self.ode_requirements = ode_requirements
        self.ODE = None
        self.reward_calc = FilterBestSpace(REWARD["State_Condition"])
        self.state_error_calc = FilterBestSpace(CONSTRAINTS)

        
    def calculate_dervative(self, segments_info, time_inputs, batch_indices_cpu):

        ODE = self.ode_requirements["calculate_dervative"]
        all_dervatives = []
        for segment in segments_info:
            torch.manual_seed(segment[0].item())
            parent = MLP(
                input_dim=int(GENERATOR["input"]),
                output_dim=int(GENERATOR["output"]),
                hidden_sizes=GENERATOR["layers"],
            ).to(DEVICE)
            
            in_par_batch = time_inputs[batch_indices_cpu[segment[1].item():segment[2].item(), 1].numpy().tolist(), :, :]
            t_input = in_par_batch.clone().detach().requires_grad_(True)
            output_for_der = parent(t_input)
            
            saved_derivatives_per_state = []
            for state_idx, state_col in enumerate(ODE['states']):
                highest_order = ODE['highest'][state_idx]
                orders_to_save = ODE['saved'][state_idx]
                current_output = output_for_der[:, :, state_col]
                derivs_by_order = {}
                current_grad = None

                for order in range(1, highest_order + 1):
                    need_for_higher_order = order < highest_order
                    current_grad = torch.autograd.grad(
                        outputs=current_output.sum(),
                        inputs=t_input,
                        create_graph=need_for_higher_order,  
                        retain_graph=True  
                    )[0]
                    
                    if order in orders_to_save:
                        derivs_by_order[order] = current_grad.detach().cpu()
                    
                    if need_for_higher_order:
                        current_output = current_grad
                    else:
                        del current_grad
                        break
                
                ordered_derivs_for_state = [derivs_by_order[order] for order in orders_to_save]
                saved_derivatives_per_state.append(ordered_derivs_for_state)
                del current_output, derivs_by_order

            del parent, output_for_der, t_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if saved_derivatives_per_state:
                all_dervatives.append(saved_derivatives_per_state)

        return all_dervatives

    def check_gradients(self, all_dervatives : list[list[list[torch.Tensor]]], segments_info):
        """
        The first list is basically related to the model number 
        The second list is related to which state it is 
        The third list is a list of tensors that we calculate the different order of dervatives with
        all_dervatives  =      [
            # Segment 0
            [
                [state_0_order_1, state_0_order_2],  # State 0
                [state_1_order_1, state_1_order_3],  # State 1 ##Based on the order we are saving in the config, `Unit test to make sure everything is in order`
            ],
            # Segment 1
            [
                [state_0_order_1, state_0_order_2],  # State 0
                [state_1_order_1, state_1_order_3],  # State 1
            ],
        ]
        """
        sorted_conditions = []
        for condition in self.ode_requirements["check_gradients"]:
            which_state = int(re.search(r'(?<=state_)\d+$', condition['state']).group()) 
            which_order = int(condition['derivative_order']) -1
            is_range = condition['type']
            target_value = float(condition['value'])
            all_values = {}
            if is_range == "standard":
                which_element = re.search(r't = (-?\d+)$',  condition['condition'])
                time_step = int(which_element.group(1))
                for i in range(len(all_dervatives)):
                    current_value = all_dervatives[i][which_state][which_order][:,time_step,:]
                    distance = torch.sqrt((current_value**2 - target_value**2).abs()) 
                    for row_idx in range(distance.shape[0]):
                        value = distance[row_idx, 0].item()
                        all_values[f"{str(segments_info[i][0].item())}_{str(row_idx + segments_info[i][1].item())}"] = value  
            else:
                for i in range(len(all_dervatives)):
                    current_value = all_dervatives[i][which_state][which_order][:,:,:]
                    penalty = 1 / (1 + torch.exp(-10*((current_value - target_value).abs())))
                    batch_avg = penalty.mean(dim=(1, 2))
                    for row_idx in range(batch_avg.shape[0]):
                        value = batch_avg[row_idx].item()  
                        all_values[f"{str(segments_info[i][0].item())}_{str(row_idx + segments_info[i][1].item())}"] = value
            sorted_conditions.append(all_values)
        result_dict = {key: 0 for key in sorted_conditions[0].keys()}
        for d in sorted_conditions:
            for key in result_dict:
                result_dict[key] += d[key]
        sorted_result_dict  = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1])}  

        return  [[int(key.split('_')[0]), int(key.split('_')[1]), value] for key, value in sorted_result_dict.items()], sorted_conditions     
            
    def calculate_nonlinearity(self, state_tensor, action_tensor=None, derivative_matrix=None, 
                            calculation_type="nonlinearity"):

        
        is_3d = state_tensor.dim() == 3
        batch_size = state_tensor.shape
        
        if calculation_type == "nonlinearity1":
            config = self.ode_requirements.get("calculate_nonlinearity", {})
        elif calculation_type == "nonlinearity2":
            config = self.ode_requirements.get("calculate_nonlinearity2", {})
        elif calculation_type == "equation":
            config = self.ode_requirements.get("equation_terms", {})
        else:
            raise ValueError(f"Unknown calculation_type: {calculation_type}")
        
        if not config:
            return {'terms': {}, 'actions': {}, 'derivatives': {}, 'scores': None}
        
        operations = {
            'sin': torch.sin, 'cos': torch.cos, 'tan': torch.tan,
            'exp': torch.exp, 'log': torch.log, 'sqrt': torch.sqrt,
            'abs': torch.abs, 'pow': torch.pow, 'sinh': torch.sinh,
            'cosh': torch.cosh, 'tanh': torch.tanh, 'square': torch.square,
            'asin': torch.asin, 'acos': torch.acos, 'atan': torch.atan,
            'ceil': torch.ceil, 'floor': torch.floor, 'round': torch.round,
            'sigmoid': torch.sigmoid, 'relu': torch.relu, 'clamp': torch.clamp,
            'random': lambda x: torch.rand_like(x),  # Handle random function
        }
        
        results = {
            'terms': {},
            'actions': {},
            'derivatives': {},
            'scores': None
        }
        
        for term_name, term_info in config.items():
            try:
                expression = term_info.get('expression', '')
                if not expression:
                    results['terms'][term_name] = None
                    continue
                
                eval_context = self._build_eval_context(
                    state_tensor, action_tensor, derivative_matrix,
                    term_info, is_3d, operations
                )
                
                result = eval(expression, {"__builtins__": {}}, eval_context)
                results['terms'][term_name] = result
                
            except Exception as e:
                print(f"Error calculating '{term_name}': {str(e)}")
                results['terms'][term_name] = None
        
        if action_tensor is not None:
            action_info = config.get('action_dependencies', [])
            for action_idx in action_info:
                key = f'action_{action_idx}'
                if action_tensor.shape[-1] > action_idx:
                    if is_3d:
                        results['actions'][key] = action_tensor[:, :, action_idx]
                    else:
                        results['actions'][key] = action_tensor[:, action_idx]
        
        if derivative_matrix is not None:
            for term_info in config.values():
                deriv_deps = term_info.get('derivative_dependencies', [])
                for state_idx, deriv_order in deriv_deps:
                    key = (state_idx, deriv_order)
                    if derivative_matrix[f'state_{state_idx}'] is not None:
                        results['derivatives'][key] = derivative_matrix[f'state_{state_idx}'][:,:,deriv_order - 1]
        
        if calculation_type == "equation":
            scores = self._calculate_equation_scores(results['terms'])
            results['scores'] = scores
        
        return results


    def _build_eval_context(self, state_tensor, action_tensor, derivative_matrix,
                        term_info, is_3d, operations):
        eval_context = {}
        state_deps = term_info.get('state_dependencies', [])
        for state_idx in state_deps:
            if state_idx < state_tensor.shape[2 if is_3d else 1]:
                if is_3d:
                    eval_context[f'state_{state_idx}'] = state_tensor[:, :, state_idx]
                else:
                    eval_context[f'state_{state_idx}'] = state_tensor[:, state_idx]
            else:
                raise IndexError(f"state_{state_idx} out of bounds")
        
        if action_tensor is not None:
            action_deps = term_info.get('action_dependencies', [])
            for action_idx in action_deps:
                if action_idx < action_tensor.shape[2 if is_3d else 1]:
                    if is_3d:
                        eval_context[f'action_{action_idx}'] = action_tensor[:, :, action_idx]
                    else:
                        eval_context[f'action_{action_idx}'] = action_tensor[:, action_idx]
                else:
                    raise IndexError(f"action_{action_idx} out of bounds")
        
        if derivative_matrix is not None:
            deriv_deps = term_info.get('derivative_dependencies', [])
            for state_idx, deriv_order in deriv_deps:
                state_key = f"state_{state_idx}"
                if derivative_matrix.get(state_key)  is not None:
                    deriv_tensor = derivative_matrix[state_key][:,:,deriv_order - 1]
                    eval_context[f'state_{state_idx}_{deriv_order}'] = deriv_tensor
        
        ident_deps = term_info.get('ident_dependencies', [])
        for ident_idx in ident_deps:
            term_key = f'term_{ident_idx}'
            if term_key in self._calculated_terms:  
                eval_context[term_key] = self._calculated_terms[term_key]
        
        eval_context.update(operations)
        return eval_context


    def _calculate_equation_scores(self, terms_dict):
        if not terms_dict:
            return None
        equations = [v for v in terms_dict.values() if v is not None]
        if not equations:
            return None
        means = [eq.mean(1) for eq in equations]
        scores = torch.stack(means, dim=0).mean(0).abs()
        return scores


    def _clear_all_gpu_memory(self):
        for ns in (globals(), locals()):
            for name, obj in list(ns.items()):
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    del ns[name]
                elif isinstance(obj, torch.nn.Module):
                    del ns[name]
        gc.collect()
        torch.cuda.empty_cache()


    def _get_derivatives_for_training_torch(self,model, time, ode_config):
        t_input = time.clone().detach().requires_grad_(True)
        output_for_der = model(t_input)
        derivative_matrix = {}
        num_states = len(ode_config.get('states', []))
        for state_idx, state_col in enumerate(ode_config['states']):
            highest_order = ode_config['highest'][state_idx]
            orders_to_save = ode_config['saved'][state_idx]
            current_output = output_for_der[:, state_col:state_col+1]  
            derivs_for_state = []
            for order in range(1, highest_order + 1):
                grad_outputs = torch.ones_like(current_output)
                current_grad = torch.autograd.grad(
                    outputs=current_output,
                    inputs=t_input,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                if order in orders_to_save:
                    derivs_for_state.append(current_grad[:, 0].unsqueeze(-1))
                if order < highest_order:
                    current_output = current_grad[:, 0:1]  
                else:
                    break
            if derivs_for_state:
                state_deriv_tensor = torch.cat(derivs_for_state, dim=1)
                derivative_matrix[f"state_{state_idx}"] = state_deriv_tensor
        return derivative_matrix

    def optimize(self, valid_state, model, time_inputs, scores, **parameters):
        def compute_derivatives_single_batch(model, t_input, ODE_cfg = self.ode_requirements["calculate_dervative"]):
            if ODE_cfg is None:
                ODE_cfg = self.ode_requirements["calculate_dervative"]
            t_input = t_input.clone().detach().requires_grad_(True)  
            output = model(t_input)                                  
            derivative_dict = {}
            for state_idx, state_col in enumerate(ODE_cfg["states"]):
                highest_order = ODE_cfg["highest"][state_idx]
                orders_to_save = ODE_cfg["saved"][state_idx]
                current_output = output[:, :, state_col]  
                collected = []
                for order in range(1, highest_order + 1):

                    need_graph = order < highest_order

                    grad = torch.autograd.grad(
                        outputs=current_output.sum(),
                        inputs=t_input,
                        create_graph=need_graph,
                        retain_graph=True
                    )[0]   
                    if order in orders_to_save:
                        dt = grad[:, :, 0]  
                        collected.append(dt.unsqueeze(-1))  
                    current_output = grad[:, :, 0]
                    if not need_graph:
                        break
                derivative_tensor = torch.cat(collected, dim=-1).cpu()    
                derivative_dict[f"state_{state_idx}"] = derivative_tensor
            return derivative_dict
        predicted_state = None
        pretrain_epochs = parameters.get('pretrain_epochs', 20000)
        train_epochs = parameters.get('train_epochs', 500000)
        mlp_lr = parameters.get('mlp_lr', 1e-3)
        derivative_loss_weight = parameters.get('derivative_loss_weight', 0.05) 
        model_save_path = parameters.get('model_save_path', 'trained_mlp.pth')
        states = valid_state.to(DEVICE)
        opt_type = REWARD["optimization"].get("type", "fix")
        mse_loss = nn.MSELoss()
        if abs(scores['state_error']) > 1e-2:
            mlp_optimizer = torch.optim.Adam(model.parameters(), lr=mlp_lr)
            print("--- Starting Pre-training Phase (State Fitting) ---")
            model.train()
            for epoch in range(pretrain_epochs):
                mlp_optimizer.zero_grad()
                predicted_state = model(time_inputs.unsqueeze(dim=0))
                _, violation_counts, _ = self.state_error_calc.iterate_valid_tensor(
                    predicted_state.unsqueeze(dim=0), 
                    time_inputs.unsqueeze(dim=0).to(DEVICE)
                )
                loss = mse_loss(predicted_state, valid_state.unsqueeze(dim=0)) + violation_counts.sum() / DOMAIN['num_points'] 
                if torch.isnan(loss):
                    print(f"Warning: NaN loss in pre-training at epoch {epoch}. Stopping.")
                    return model
                loss.backward()
                mlp_optimizer.step()
                if epoch % 1000 == 0:
                    print(f"Pre-train Epoch {epoch}, State Prediction Loss: {loss.item():.6f}")
            print("--- Pre-training Finished ---")
            print("--- Starting Joint Optimization Phase ---")
            state = model(time_inputs.unsqueeze(dim=0)).detach().unsqueeze(-1)
        print(f"Using derivative_loss_weight = {derivative_loss_weight}")
        
        if opt_type in ("Max", "Min"):
            T_total = torch.tensor(1.0, requires_grad=True)
            optimizer = torch.optim.Adam([T_total], lr=0.1)
            prev_T_total = T_total.item()
        convergence_tol = 1e-3  
        patience = 500         
        stable_counter = 0
        
        
        if opt_type in ("Max", "Min"):
            for epoch in range(train_epochs):
                optimizer.zero_grad()
                derivatives_dict, dt = self.dependent_actions(T_total, states, order=2)
                base_loss = self.derivative_loss(derivatives_dict) 
                if opt_type == "Min":
                    loss =  base_loss + T_total
                else:
                    loss = T_total * base_loss - T_total
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    T_total.data = torch.clamp(T_total.data, min=1e-4)
                delta_T = abs(T_total.item() - prev_T_total)
                if delta_T < convergence_tol:
                    stable_counter += 1
                    if stable_counter >= patience:
                        print(f"Converged at epoch {epoch}, T_total = {T_total.item():.6f}")
                        break
                else:
                    stable_counter = 0
                prev_T_total = T_total.item()
                if epoch % 200 == 0:
                    print(epoch, "T_total =", T_total.item(), "dervatives =", base_loss.item())
            predicted_state = states
        else:
            model.train()
            B = 1  
            T = time_inputs.shape[0] 
            num_actions = DOMAIN['actions']
            
            unconstrained_actions = torch.zeros(
                (B, T, num_actions), 
                requires_grad=True, 
                device=DEVICE
            )
            optimizer = torch.optim.Adam([
                {'params': model.parameters(), 'lr': 1e-3},  
                {'params': [unconstrained_actions], 'lr': 3e-3}  
            ])
            for epoch in range(train_epochs):
                if epoch == 0:
                    best_loss = float('inf')
                    epochs_no_improve = 0
                    patience = 1000      
                    min_delta = 1e-5
                optimizer.zero_grad()
                predicted_state = model(time_inputs.unsqueeze(dim=0))
                
                reordered_derivatives = compute_derivatives_single_batch(model, time_inputs.unsqueeze(dim=0))
                reordered_derivatives_device = {key: value.to(DEVICE) for key, value in reordered_derivatives.items()}
                self._calculated_terms = {}
                nl1 = self.calculate_nonlinearity(predicted_state, calculation_type="nonlinearity1")
                self._calculated_terms = nl1["terms"]
                nl2 = self.calculate_nonlinearity(predicted_state, calculation_type="nonlinearity2")
                self._calculated_terms.update(nl2["terms"])
                action_tensor = torch.nn.functional.softplus(unconstrained_actions)
                eq_out = self.calculate_nonlinearity(
                    predicted_state.to(DEVICE),
                    action_tensor.to(DEVICE),
                    derivative_matrix=reordered_derivatives_device,
                    calculation_type="equation"
                )
                reward = self.reward_calc.expected_reward(predicted_state, time_inputs.unsqueeze(dim=0))
                state_error = mse_loss(predicted_state, valid_state.unsqueeze(dim=0).to(DEVICE))
                
                total_loss = eq_out["scores"] + 0.1*state_error - reward/DOMAIN['num_points'] 
                derivative_loss_val = 0.0
                if len(self.ode_requirements["check_gradients"]) > 0:
                    derivative_matrix = self._get_derivatives_for_training_torch(
                        model,
                        time_inputs,
                        self.ode_requirements["calculate_dervative"]
                    )
                    derivative_loss_val = self.derivative_loss(derivative_matrix)
                    total_loss += derivative_loss_val * derivative_loss_weight
                
                if torch.isnan(total_loss):
                    print(f"Warning: NaN loss at epoch {epoch}. Stopping.")
                    break
                total_loss.backward()
                with torch.no_grad():
                    if unconstrained_actions.grad is not None:
                        grad = unconstrained_actions.grad.reshape(B, -1)
                        norms = torch.norm(grad, dim=1, keepdim=True).clamp(min=1e-8)
                        grad = grad / norms
                        unconstrained_actions.grad = grad.reshape(B, T, num_actions)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                current_loss = total_loss.item()
                if epoch % 1000 == 0:
                    deriv_val = (
                        derivative_loss_val if isinstance(derivative_loss_val, (int, float)) 
                        else derivative_loss_val.item()
                    )
                    min_action = action_tensor.min().item()
                    max_action = action_tensor.max().item()
                    print(
                        f"Train Epoch {epoch}, Total Loss: {current_loss:.6f}, "
                        f"feasibility {eq_out['scores'].item()}, "
                        f"Total Reward {reward/DOMAIN['num_points']}, "
                        f"Derivative Loss: {deriv_val:.6f}, "
                        f"Action range: [{min_action:.6f}, {max_action:.6f}]"
                    )

                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
                        break

                if eq_out["scores"].item() < 1e-3:
                    print(f"Early stopping at epoch {epoch}.")
                    print('=====================Optimization Results==================================')
                    print(
                        f"Train Epoch {epoch}, Total Loss: {current_loss:.6f}, "
                        f"feasibility {eq_out['scores'].item() }, "
                        f"Total Reward {reward/DOMAIN['num_points']}, "
                        f"Derivative Loss: {deriv_val:.6f}, "
                        f"Action range: [{min_action:.6f}, {max_action:.6f}]"
                    )
                    break
        return predicted_state.detach()

    def _clear_all_gpu_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def dependent_actions(self, T_total, states, order=2):
        Tn, D = states.shape
        dt = torch.abs(T_total) / (Tn - 1)
        deriv_prev = (states[1:] - states[:-1]) / dt 
        derivatives_dict = {f"state_{i}": [] for i in range(D)}
        for i in range(D):
            derivatives_dict[f"state_{i}"].append(deriv_prev[:, i])
        for k in range(2, order + 1):
            deriv_k = (deriv_prev[1:] - deriv_prev[:-1]) / dt  
            for i in range(D):
                derivatives_dict[f"state_{i}"].append(deriv_k[:, i])
            deriv_prev = deriv_k
        return derivatives_dict, dt

    def derivative_loss(self, derivatives_dict):
        total_error = 0
        num_conditions = len(self.ode_requirements["check_gradients"])
        if num_conditions == 0:
            device = next(iter(derivatives_dict.values())).device
            return torch.tensor(0.0, device=device)
        for condition in self.ode_requirements["check_gradients"]:
            which_state = condition['state'] 
            which_order = int(condition['derivative_order']) - 1  
            is_range = condition['type']
            target_value = float(condition['value'])
            if which_state not in derivatives_dict:
                raise ValueError(f"State {which_state} not found in derivatives_dict")
            state_tensor = derivatives_dict[which_state]
            if is_range == "standard":
                time_step = int(re.search(r't = (-?\d+)$', condition['condition']).group(1))
                derivative_value = state_tensor[which_order][time_step]
                error = torch.sqrt((derivative_value**2 - target_value**2).abs())
                total_error += error
            else:
                derivative_values = state_tensor[which_order]
                error = torch.relu(torch.norm(derivative_values, dim=0) - target_value).mean()
                total_error += error 
        return total_error

    def optimize_actions(
        self,
        state_tensor,           
        derivative_matrix,     
        num_actions,
        lr=1e-2,
        steps=20000,
        verbose=True,
        patience=50,              
        min_improvement=1e-4,    
        check_interval=10
    ):
        DEVICE = state_tensor.device
        unbatched = False
        if state_tensor.dim() == 2:  
            unbatched = True
            state_tensor = state_tensor.unsqueeze(0)   
        B, T, D = state_tensor.shape
        unconstrained_actions = torch.zeros(
            (B, T, num_actions), 
            requires_grad=True, 
            device=DEVICE
        )
        optimizer = torch.optim.Adam([unconstrained_actions], lr=lr)
        best_loss_per_batch = torch.full((B,), float('inf'), device=DEVICE)
        steps_without_improvement = torch.zeros(B, dtype=torch.int32, device=DEVICE)
        converged_batches = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        for step in range(steps):
            optimizer.zero_grad()
            action_tensor = torch.nn.functional.softplus(unconstrained_actions)

            eq_out = self.calculate_nonlinearity(
                state_tensor,
                action_tensor,
                derivative_matrix=derivative_matrix,
                calculation_type="equation"
            )
            scores = eq_out["scores"]
            total_loss = torch.mean(scores)
            total_loss.backward()
            with torch.no_grad():
                if unconstrained_actions.grad is not None:
                    grad = unconstrained_actions.grad.reshape(B, -1)
                    norms = torch.norm(grad, dim=1, keepdim=True).clamp(min=1e-8)
                    grad = grad / norms
                    unconstrained_actions.grad = grad.reshape(B, T, num_actions)
            optimizer.step()
            if step % check_interval == 0:
                with torch.no_grad():
                    relative_improvement = (
                        best_loss_per_batch - scores
                    ) / (best_loss_per_batch.abs() + 1e-8)
                    for b in range(B):
                        if not converged_batches[b]:
                            if relative_improvement[b] > min_improvement:
                                best_loss_per_batch[b] = scores[b]
                                steps_without_improvement[b] = 0
                            else:
                                steps_without_improvement[b] += check_interval
                                if steps_without_improvement[b] >= patience:
                                    converged_batches[b] = True
                                    if verbose:
                                        print(f"  → Batch {b} converged at step {step} "
                                            f"(loss: {scores[b].item():.6f})")
                    if converged_batches.all():
                        if verbose:
                            print(f"All batches converged at step {step}")
                        break
            if verbose and step % 200 == 0:
                num_converged = converged_batches.sum().item()
                min_action = action_tensor.min().item()
                print(
                    f"converged: {num_converged}/{B} | min_action: {min_action:.6f}"
                )
        with torch.no_grad():
            final_actions = torch.nn.functional.softplus(unconstrained_actions)
        if unbatched:
            return final_actions[0].detach(), scores[0].detach()
        return final_actions.detach(), scores.detach()
