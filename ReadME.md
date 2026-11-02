tech node : 16nm_LP
VDD : 0.9
Testing * - bit parity checker 
To generate data : python .\src\spice\run_datatset.py

Models we are trying out :
Model	             Why try it?
Random Forest	     Reliable baseline; handles nonlinearities
Extra Trees	         Often effective for complex transistor behavior
Gradient Boosting	 Can model nonlinear relationships accurately
MLP Neural Network	 Can learn smooth nonlinear functions

Data Split Used : 70+15+15 (train_models.py[contains splitting, training and selection])

IMPORTANT ( should include in the report)
The relative error calculation uses a small denominator floor to avoid division by zero. Therefore, it is a floor-adjusted relative error, not an unmodified percentage error. Mention this definition in your report.

Also, your project requires the test-set results to be reported only after model selection on validation data. This script follows that order.

One final note: This script saves the models with their current transformation and feature information. Your teammate can load them later to predict transistor currents, without retraining.

Used P95 error and here are the final chosen model:
1. NMOS_ON = Random Forest 
2. NMOS_OFF = Extra Trees 
3. PMOS_ON = Gradient Boosting 
4. PMOS_OFF = MLP