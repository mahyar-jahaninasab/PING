import torch


class FilterBestSpace:

    def __init__(self, conditions: list[dict]):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conditions = conditions

    def mask_variable(self, 
                      variable: torch.Tensor, 
                      logic: dict, 
                      column: int = None):

        if not isinstance(variable, torch.Tensor):
            variable = torch.tensor(variable, 
                                    device=self.device)
        
        if column is not None:
            variable = variable[:, :, column]
        
        mask = torch.ones(variable.shape, 
                          dtype=torch.bool, 
                          device=self.device)
        
        if 'lower' in logic:
            lower_val = logic['lower']
            value, op = lower_val
            if op == '<=':
                mask &= (variable >= value)
            elif op == '<':
                mask &= (variable > value)
            else: 
                raise ValueError(f"Invalid lower operator: {op}. The mask logic only supports '<' or '<='")

        if 'upper' in logic:
            upper_val = logic['upper']
            value, op = upper_val
            if op == '<=':
                mask &= (variable <= value)
            elif op == '<':
                mask &= (variable < value)
            else: 
                raise ValueError(f"Invalid upper operator: {op}. The mask logic only supports '<' or '<='")    

        return mask
    
    def iterate_over_constrains(self, 
                                state_variables: torch.Tensor, 
                                input_variables: torch.Tensor):

        for enum, constrain in enumerate(self.conditions):
            input_constrain = constrain['time']
            state_constrains = constrain['states']
            result_and = self.mask_variable(input_variables, input_constrain)
            result_and = result_and.repeat(state_variables.shape[0]//input_variables.shape[0], 1, 1)
            masks = []
            for idx, (_, state_constrain) in enumerate(state_constrains.items()):
                mask = self.mask_variable(state_variables, state_constrain, idx).unsqueeze(-1)
                masks.append(mask)
            for mask in masks:
                result_and = torch.logical_and(result_and, mask)
            
            if enum == 0:
                result_or = result_and
            else:
                result_or = torch.logical_or(result_or, result_and)
        return ~result_or

    def filter_valid_tensor(self, 
                            state_variables: torch.Tensor,
                            input_variables: torch.Tensor,
                            fallback_percentage: float = 0.4) -> torch.Tensor:
        
        result_flags = self.iterate_over_constrains(state_variables, input_variables)
        false_counts = (~result_flags).sum(dim=(1, 2))
        min_false_count = false_counts.min()
        min_violation_indices = torch.nonzero(false_counts == min_false_count, as_tuple=True)[0]
        num_to_return = max(1, int(state_variables.shape[0] * fallback_percentage))

        if len(min_violation_indices) > num_to_return:
            perm = torch.randperm(len(min_violation_indices), device=state_variables.device)
            best_indices = min_violation_indices[perm[:num_to_return]]
            best_indices = torch.sort(best_indices)[0]

        else:
            best_indices = min_violation_indices
        violation_counts = false_counts[best_indices]

        best_mask = torch.zeros(state_variables.shape[0], dtype=torch.bool, device=state_variables.device)
        best_mask[best_indices] = True
        return state_variables[best_mask], best_indices, violation_counts 

    def iterate_valid_tensor(self, 
                             state_variables: torch.Tensor, 
                             input_variables: torch.Tensor) -> torch.Tensor:

        if state_variables.numel() == 0:
            empty_valids = torch.empty(
                0,
                state_variables.shape[2],
                state_variables.shape[3],
                dtype=state_variables.dtype,
                device=state_variables.device,
            )
            empty_indices = torch.empty(0, dtype=torch.long, device= state_variables.device)
            empty_batch_indices = torch.empty(0, 2, dtype=torch.long, device= state_variables.device)
            return empty_valids, empty_indices, empty_batch_indices
        B, N, M, D = state_variables.shape
        flat = state_variables.reshape(B * N, M, D)
        valid_states, flat_indices, violation_counts = self.filter_valid_tensor(flat, input_variables)
        model_indices = flat_indices // N  
        valid_indices = flat_indices % N   
        model_batch_indices = torch.stack([model_indices, valid_indices], dim=1)
        return valid_states, violation_counts, model_batch_indices
    

    def expected_reward(self, 
                       state_variables: torch.Tensor, 
                       input_variables: torch.Tensor) -> torch.Tensor:
        
        result_flags = self.iterate_over_constrains(state_variables, input_variables)
        reward_counts = (~result_flags).sum(dim=(1, 2))      
        time_interval = self.conditions[0].get("time", {})
        def _get_time(bound: str, default_tensor: torch.Tensor) -> torch.Tensor:
            value = time_interval.get(bound)
            if value is not None:
                return default_tensor.new_full(default_tensor.shape, value[0])
            return default_tensor
        starting_time = _get_time("lower", input_variables[:, 0, :].flatten())
        ending_time   = _get_time("upper", input_variables[:, -1, :].flatten())  
        delta_t = ending_time - starting_time
        return delta_t * reward_counts
    

