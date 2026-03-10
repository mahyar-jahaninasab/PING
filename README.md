# PING


## Overview
PING is a training-free framework that uses untrained neural networks and symbolic verification for rapid, high-fidelity trajectory generation. Designed for continuous-time planning domains, it outperforms traditional optimization baselines in both speed and scalability. By decoupling generation from optimization, PING circumvents discretization artifacts and avoids local minima traps. 
## License and Citation

This project and its original source code are licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
```
**Academic Use:**
If you use this framework, code, or the PING/PING+ methodology in your research, we ask that you fulfill your academic obligation by citing the following paper:

bibtex
@inproceedings{ping2026,
  title={PING: A Physics-Informed Neuro-Symbolic Generator for Continuous-Time Planning},
  author={Mahyar Jahani-nasab, Hamid Rezatofighi, Mor Vered, Buser Say},
  booktitle={Proceedings of the International Conference on Automated Planning and Scheduling (ICAPS)},
  year={2026},
  note={Accepted}
}
```
## Key Features
- **Training-Free Synthesis**: Leverages untrained neural networks as generative function approximators without requiring prior data or offline training.
- **Symbolic Verification**: Uses automatic differentiation to rigorously validate generated candidate trajectories against domain-specific differential equations.
- **Continuous-Time Natively**: Operates directly in continuous function-valued spaces to strictly satisfy boundary conditions.

## Supported Domains
The framework is highly effective for complex, high-dimensional continuous domains

## Installation
### Clone the repository
git clone https://github.com/mahyar-jahaninasab/ping.git

### Navigate into the project directory
cd PING

### Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

### Install the project and dependencies
pip install -e .

### Run the pipeline
python pipeline.py



