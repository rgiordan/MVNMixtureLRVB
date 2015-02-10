# MVNMixtureLRVB

Install using 

`library(devtools)`
`install_github("rgiordan/MVNMixtureLRVB")`

Apologies for the current poor state of the documentation.  An example comparison between
Gibbs and LRVB can be found in `inst/simulate_multivariate_normal_mixture_sandbox.R`.
Note that the VB fitting currently uses flat priors, so it will crash if any component
ends up with zero data assigned to it.


