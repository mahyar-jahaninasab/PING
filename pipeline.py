import os
from dotenv import load_dotenv
from parser import ODEParser, DomainConfiguration, create_config
load_dotenv()
DESCRIPTION = os.getenv("DESCRIPTION")

with open(DESCRIPTION, "r") as f:
    ODE_TEXT = f.read()


parser = ODEParser()
parser.parse_ode_text(ODE_TEXT)

domain               = parser.get_config_by_section('#Domain')
constraints          = parser.get_config_by_section('#Constraints')
reward               = parser.get_config_by_section('#Reward')
ode_system           = parser.get_config_by_section('#ODE_System')
derivative_constraints = parser.get_config_by_section('#Derivative_Value_Constraints')
nonlinear_terms      = parser.get_config_by_section('#NonLinear_Terms')
nonlinear_terms2     = parser.get_config_by_section('#NonLinear_Terms2')
equation_terms       = parser.get_config_by_section('#Equations')


config = {
    "DOMAIN": domain,
    "CONSTRAINTS": constraints,
    "REWARD": reward,
    "ODE": ode_system,
    "DERIVATIVES": derivative_constraints,
    "NONLINEAR_TERMS": nonlinear_terms,
}
domain_cfg = DomainConfiguration(config)
config = domain_cfg.pipeline()


data = {
    "DOMAIN": config["DOMAIN"],
    "CONSTRAINTS": config["CONSTRAINTS"],
    "REWARD": config["REWARD"],

    "GENERATOR": {
        "input": 1,
        "output": config["DOMAIN"]["states"],
        "layers": [config["DOMAIN"].get("nodes", 128)] * config["DOMAIN"].get("layers", 4),
        "numbers": config["DOMAIN"].get("numbers", 100),
        "start":  config["REWARD"]["Initial_Condition"],
        "goal": config["DOMAIN"]["target_values"],
        "batch_size": config["DOMAIN"].get("batch_size", 100),
        "num_points": config["DOMAIN"].get("num_points", 1000),
        "upper_bound_time": config["DOMAIN"].get("upper_bound_time", False),
        "descending": config["DOMAIN"].get("descending", None),
        "rate": config["DOMAIN"].get("rate", 0.1),
    },

    "TRAINING": {
        "goal_oriented": config["DOMAIN"]["goal_oriented"]
    }
}

create_config(data)


ode_requirements = {
    "calculate_dervative": ode_system["calculate_dervative"],
    "check_gradients": derivative_constraints,
    "calculate_nonlinearity": nonlinear_terms,
    "calculate_nonlinearity2": nonlinear_terms2,
    "equation_terms": equation_terms,
}

def pipeline():
    import numpy as np
    import torch
    import json
    from manager import MLP
    from filtering import FilterBestSpace
    from ode import odeFeasibility
    from post_processing import SolutionSelector, reorder_derivatives_by_state
    from manager import ForwardPhase
    from dotenv import load_dotenv
    import time
    from plot import plot_icaps_figure


    load_dotenv()
    problem_path = os.getenv("PROBLEM")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(problem_path) as f:
        data = json.load(f)

    DOMAIN      = data["DOMAIN"]
    GENERATOR   = data["GENERATOR"]
    CONSTRAINTS = data["CONSTRAINTS"]
    REWARD      = data["REWARD"]

    COMPONENTS = {
        "IC": len(REWARD["Initial_Condition"]) > 0,
        "DERIVATIVES": len(derivative_constraints) > 0,
        "DURATION_MAX": REWARD["optimization"].get("type", None) == "Max",
        "DURATION_MIN": REWARD["optimization"].get("type", None) == "Min",
        "STATE": len(CONSTRAINTS) > 0,
        "REWARD": len(REWARD["State_Condition"]) > 0,
        "ACTIONS": DOMAIN["actions"] > 0,
        "EQUATION_NONLINEAR_TERMS_1": len(nonlinear_terms) > 0,
        "EQUATION_NONLINEAR_TERMS_2": len(nonlinear_terms2) > 0,
        "EQUATION": len(equation_terms) > 0,
    }

    constraint_validator = FilterBestSpace(CONSTRAINTS)

    forward_model = ForwardPhase()
    t0 = time.time()
    model_outputs, time_inputs, metadata = forward_model.generate()
    t_generation = time.time() - t0

    stacked_outputs = torch.stack(model_outputs, dim=0)
    t1 = time.time()
    valid_states, violation_counts, batch_indices = constraint_validator.iterate_valid_tensor(
        stacked_outputs, 
        time_inputs.to(DEVICE)
    )
    torch.set_printoptions(profile='full')
    t_filtering_1 = time.time() - t1

    valid_states_cpu = valid_states.cpu()
    batch_indices_cpu = batch_indices.cpu()

    model_indices = batch_indices_cpu[:, 0]
    transitions = torch.nonzero(model_indices[1:] != model_indices[:-1]).flatten() + 1

    boundaries = torch.cat([
        torch.tensor([0]),
        transitions,
        torch.tensor([len(model_indices)])
    ])

    segments_info = torch.stack([
        model_indices[boundaries[:-1]],
        boundaries[:-1],
        boundaries[1:]
    ], dim=1)

    PINN = odeFeasibility(ode_requirements)
    t_2 = time.time()

    derivatives = PINN.calculate_dervative(segments_info, time_inputs, batch_indices_cpu)
    reordered_derivatives = reorder_derivatives_by_state(derivatives)

    if COMPONENTS["DERIVATIVES"]:
        print(COMPONENTS["DERIVATIVES"])
        gradient_score, all_gradient_score = PINN.check_gradients(derivatives, segments_info)
        gradient_score_sorted = np.array(sorted(gradient_score, key=lambda x: x[1]))[:, -1].astype(float)
        
    else:
        gradient_score_sorted = None

    if COMPONENTS["REWARD"]:
        print(COMPONENTS["REWARD"])
        reward_calc = FilterBestSpace(REWARD["State_Condition"])
        state_reward = reward_calc.expected_reward(valid_states, time_inputs).detach().cpu().numpy()
    else:
        state_reward = None

    if COMPONENTS["DURATION_MAX"] and COMPONENTS["DURATION_MIN"]:
        print("reward_type")
        time_reward = time_inputs[batch_indices_cpu[:, 1]][:, -1, 0].cpu().numpy().astype(float)
    else:
        time_reward = None

    if COMPONENTS["EQUATION"]:
        if COMPONENTS["EQUATION_NONLINEAR_TERMS_1"]:
            nl1 = PINN.calculate_nonlinearity(valid_states_cpu.to(DEVICE), calculation_type="nonlinearity1")
            PINN._calculated_terms = nl1["terms"]

        if COMPONENTS["EQUATION_NONLINEAR_TERMS_2"]:
            nl2 = PINN.calculate_nonlinearity(valid_states_cpu.to(DEVICE), calculation_type="nonlinearity2")
            PINN._calculated_terms.update(nl2["terms"])
        action_output, equation_scores = PINN.optimize_actions(valid_states_cpu.to(DEVICE),reordered_derivatives,DOMAIN['actions'])
        equation_scores = equation_scores.cpu().numpy()
    else:
        equation_scores = None

    selector = SolutionSelector()
    selection = selector.select(
        violation_counts.cpu().numpy().astype(float),
        time_reward,
        equation_scores,
        None,
        state_reward
    )

    t_filtering_2 = time.time() - t_2
    t_filtering = t_filtering_1 + t_filtering_2

    torch.manual_seed(batch_indices_cpu[selection[0]][0])
    if reordered_derivatives is not None:
        chosen_derivatives = {}
        for key, tensor in reordered_derivatives.items():
            chosen_derivatives[key] = tensor[selection[0], :, :]


    model =  MLP(
                input_dim= int(GENERATOR["input"]),
                output_dim= int(GENERATOR["output"]),
                hidden_sizes= GENERATOR["layers"],
                parent=True
            ).to(DEVICE)



    t_3 = time.time()
    output = PINN.optimize(valid_states_cpu[selection[0]].to(DEVICE),
                    model,
                    time_inputs[batch_indices_cpu[:, 1].tolist()].detach()[selection[0]].to(DEVICE),
                    selection[1],
                    )
    t_improvement = time.time() - t_3


    print("\n=== Example completed! ===")
    print(len(valid_states_cpu))

    print(
        f"\n===== PIPELINE SUMMARY =====\n"
        f"Generation time     : {t_generation:.4f} seconds\n"
        f"Filtering time      : {t_filtering:.4f} seconds\n"
        f"Optimization time   : {t_improvement:.4f} seconds\n"
        f"Total runtime       : {t_generation + t_filtering + t_improvement:.4f} seconds\n"
        f"Number of Zero shot solutions: {len(valid_states_cpu)}\n"
        f"Filtering result : {selection}\n"
        f"each trajectory: {(t_generation + t_filtering + t_improvement)/len(valid_states_cpu)}\n"
        f"============================\n"
    )

    print("\n=== Example completed! ===")

if __name__ == "__main__":
    pipeline()

    