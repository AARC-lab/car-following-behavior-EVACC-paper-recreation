# GM First-Order Car-Following Model
General Motors (GM) First-Order Model, a linear stimulus-response model. It models how a group of vehicles react to the speed changes of a leading vehicle over time.

## Model Equation
The acceleration of a following vehicle is proportional to the velocity difference between itself and the preceding vehicle:  

$$
a_n(t) = \lambda \left[ v_{n-1}(t - \tau) - v_n(t - \tau) \right]

$$
where:  

- $ a_n(t) $ → Acceleration of the **n-th** vehicle at time \( t \).  
-  $\lambda$ → Sensitivity coefficient (stimulus coefficient).  
- $v_n(t)$ → Speed of the **n-th** vehicle at time \( t \).  
- $v_{n-1}(t)$ → Speed of the **leading vehicle** at time \( t \).  
- $ \tau $ → Reaction time delay.  
