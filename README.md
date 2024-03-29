# PostoMETRO-Paper
### [Project Page](https://postometro.github.io/) 

- This is a repo for our paper :point_right: **PostoMETRO: Pose Token Enhanced Mesh Transformer for Robust 3D Human Mesh Recovery**

## Visualization :eyes:

<!-- ![Teaser 1](./assets/teaser_narrow_1.gif) ![Teaser 2](./assets/teaser_wide_1.gif) -->
We offer some GIF teasers to demonstrate the generalizability of PostoMETRO

<div class="container is-max-desktop">
  <div class="columns is-centered has-text-centered">
    <div class="column is-full-width">
      <div style="display: flex; align-items: center;">
        <img src="./assets/teaser_narrow_1.gif" alt="teaser 1" style="height: 330px; margin-right: 10px; margin-bottom: 20px">
        <img src="./assets/teaser_wide_1.gif" alt="teaser 2" style="height: 330px; margin-bottom: 20px">
      </div>
      <div style="display: flex; align-items: center;">
        <img src="./assets/teaser_narrow_2.gif" alt="teaser 3" style="height: 330px; margin-right: 10px; margin-bottom: 20px">
        <img src="./assets/teaser_wide_2.gif" alt="teaser 4" style="height: 330px; margin-bottom: 20px">
      </div>
    </div>
  </div>
</div>

We also offer visualization results on 3DPW and OCHuman datasets, and occlusion sensitivity analysis results compared with other baselines

<div align="center">
  <img src="./assets/vis1.png" alt="vis">
</div>


<div align="center">
  <img src="./assets/occ_sens.png" alt="occ-sens">
</div>

For more results, check our paper!

## Overview :monocle_face:

With the recent advancements in single-image-based human mesh recovery, there is a growing interest in enhancing its performance in certain extreme scenarios, such as occlusion, while maintaining overall model accuracy. Although obtaining accurately annotated 3D human poses under occlusion is challenging, there is still a wealth of rich and precise 2D pose annotations that can be leveraged. However, existing works mostly focus on directly leveraging 2D pose coordinates to estimate 3D pose and mesh. In this paper, we present **PostoMETRO** (**Pos**e **to**ken enhanced **ME**sh **TR**ansf**O**rmer), which integrates occlusion-resilient 2D pose representation into transformers in a token-wise manner. Utilizing a specialized pose tokenizer, we efficiently condense 2D pose data to a compact sequence of pose tokens and feed them to the transformer together with the image tokens. This process not only ensures a rich depiction of texture from the image but also fosters a robust integration of pose and image information. Subsequently, these combined tokens are queried by vertex and joint tokens to decode 3D coordinates of mesh vertices and human joints. Facilitated by the robust pose token representation and the effective combination, we are able to produce more precise 3D coordinates, even under extreme scenarios like occlusion. Experiments on both standard and occlusion-specific benchmarks demonstrate the effectiveness of **PostoMETRO**. Qualitative results further illustrate the clarity of how 2D pose can help 3D reconstruction. Code will be made available.

<div align="center">
  <img src="./assets/paradigm.png" alt="Overview Image">
</div>


<div align="center">
  <img src="./assets/model.png" alt="Overview Image">
</div>

## Result :rocket:

We also some offer quantitive results for better comparison

<div align="center">
  <img src="./assets/tab1.png" alt="result1">
  <img src="./assets/tab2.png" alt="result2">
</div>

For more results, check our paper!