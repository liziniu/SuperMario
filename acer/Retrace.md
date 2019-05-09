## Retrace Algorithm

The retrace update for target q-value is :
$$
q_{target_t} = r_t + \gamma * (\bar \rho_t * (q_{target_{t+1}} -q_{t+1} ) + v_{t+1})
$$
For the initial value, 
$$
q_{target_{t+1}} = q_{t+1}
$$
The naive acer programe will compute the $q_{t+1}$ and $v_{t+1}$ in the following way:
$$
q_{t+1} = f(o_{t+1}, g_{t+1})
$$

$$
v_{t+1} = \sum_{a_{t+1}} \pi({a_{t+1}} | o_{t+1}, g_{t+1})q_{t+1}
$$

Note that for $(s_t, g_t, a_t, s_{t+1}, g_{t+1})$, we require that $g_t = g_{t+1}$ to satisfy that goal is invariant in a trajectory.

For the computation of acer, we expect that:
$$
q_{t+1} = f(o_{t+1}, g_{t})
$$

$$
v_{t+1} = \sum_{a_{t+1}} \pi({a_{t+1}}|o_{t+1}, g_t)q_{t+1}
$$

