## Auxilliary Tasks

Auxilliary tasks are designed as a part of the dynamics model to extract image features. Three types of auxilliary tasks are considered:

- ### RandomFeature

Use a randomly initialized and fixed (fixed meaning no training) network to extract features.

- ### InverseDynamics

Use a inversed model to extract features by recovering actions based on observations and next observations.

- ### RandomNetworkDistillation

Use a randomly initialized but trainbale network to extract features by approximating a randomly initialized and fixed network.



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