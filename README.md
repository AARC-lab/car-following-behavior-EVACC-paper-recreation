# GM First-Order Car-Following Model  

The **General Motors (GM) First-Order Model** is a linear **stimulus-response** model that describes how a group of vehicles react to the speed changes of a leading vehicle over time.  

## Model Equation  

The acceleration of a following vehicle is proportional to the velocity difference between itself and the preceding vehicle:  

$$
a_n(t) = \lambda \left[ v_{n-1}(t - \tau) - v_n(t - \tau) \right]
$$

where:  

- **\( a_n(t) \)** → Acceleration of the **n-th** vehicle at time \( t \).  
- **\( \lambda \)** → Sensitivity coefficient (stimulus coefficient).  
- **\( v_n(t) \)** → Speed of the **n-th** vehicle at time \( t \).  
- **\( v_{n-1}(t) \)** → Speed of the **leading vehicle** at time \( t \).  
- **\( \tau \)** → Reaction time delay.  

## Description  

This model assumes that a following vehicle adjusts its acceleration based on the speed difference between itself and the preceding vehicle after a reaction time delay (\( \tau \)). The **sensitivity coefficient (\( \lambda \))** determines how aggressively the following vehicle reacts to speed changes in the lead vehicle.  

## Usage  

The GM First-Order Model is widely used in:  

- **Traffic Flow Simulation** 🏎️  
- **Autonomous Vehicle Control** 🤖  
- **Driver Behavior Analysis** 🚗💡  

---

📘 *For more details, refer to traffic flow theory resources and car-following model studies.*
