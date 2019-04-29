## Acer 

There are some differences with the origin acer procedure.

1. We require ``eval_env``  since exploration policy and evaluation policy are distinct except that the ``dynamics`` feed to exploration policy is ``DummyDynamics`` . On that situation, the evluation policy is the same with expolration policy. If ``eval_env`` is None, we will create a env with number = 1. 
2. We require ``dynamics``, which is feed to exploration policy. If you want to use origin acer, please set ``dynamics``  to ``DummyDynamics``.
3.  We call ``acer.initialize()``, which collects goals before training starts. 
4. We call ``acer.evaluation()``, which use evaluation policy to evaluate. 

## Model

This module is for training acer policy and dynamics. If there is no dynamics (like evaluation policy), please use the ``DummyDynamics`` and will set ``goal_placeholder`` useless.

```
training_policies
```

Same with origin acer except that there is a ``goal_feat `` input, which is concated with ``policy_latent`` to ouput the policy. 

```
training_dynamics
```

Use ``obs、actions、next_obs`` to train dynamics model. This function is only called when collect samples by on-policy.



## Runner

- ### Methods

```
1. initialize
```

  Runner for ``init_step``  to collect ``obs`` , and make sure we can ``get_goal`` at the first stage.

```
2. run
```

The diagram for ``run`` is summarized as:

1. Sample ``goal_obs`` and recompute ``goal_feat``

2. For $t$ = 1: T
3. ​       If  ``reached_goal[env_id]``
4. ​               actions = random_action();
5. ​      else  
6. ​               actions = sample_policy.step(``obs``,  ``goal_feat``)
7. ​       env.step(actions)

Note that  we resample goal when ``run``  is called.

```
3. check_reached
```

Process the current obs into embedding space and then calculate distance based on L2-distance.



## Buffer

- ### Methods

```
1. put
```

 Store ```enc_obs``` , ``actions`` , ``rewards`` , ``mus``, ``dones``, ``maks``, ``goal_obs``.

Note that we store ``goal_obs`` rather than ``goal_feat`` , because we will recompute ``goal_feat`` when it is selected to training.

```
2. get
```

Take out data we have stored. 

It requries ``sample_goal_fn`` , which decides how to replace current goal with future goal. 

What's more, we will recompute ``int_rewards`` , which meaning that we process ``obs`` and ``goa_obs`` into feature embedding and make the negative distance as ``int_rewards``.