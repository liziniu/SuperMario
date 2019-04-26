## Dynamics:

- ### Methods

```
1. extract_feature
```

Process ``obs``  into ``obs_feat``.

```
2. get_goal
```

Take out the goals(``obs``  and ``obs_feat``) with the most high priority.

Note: we recompute ``obs_feat`` when take out ``obs``

```
3. put_goal
```

Store the ``obs`` into the priority queue. priority is measured by ``dynamics.novelty`` 

- ### Training

  #### Loss

  - ``aux_loss``:  loss from the model that process image into feature embedding.  For ``RandomFeature`` , it is 0.

  - ``dyna_loss``: loss from the model that measure the novelty of embedding.  For ``RandomNetworkDistillation`` , it is 0. 

  #### Feed

  1. ``obs``
  2. ``actions``
  3. ``next_obs``

  

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

## Model





