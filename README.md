# PostoMETRO-Paper

## Overview

With the recent advancements in single-image-based human mesh recovery, there is a growing interest in enhancing its performance in certain extreme scenarios, such as occlusion, while maintaining overall model accuracy. Although obtaining accurately annotated 3D human poses under occlusion is challenging, there is still a wealth of rich and precise 2D pose annotations that can be leveraged. However, existing works mostly focus on directly leveraging 2D pose coordinates to estimate 3D pose and mesh. In this paper, we present $\textbf{PostoMETRO}$ ($\textbf{Pos}$e $\textbf{to}$ken enhanced $\textbf{ME}$sh $\textbf{TR}$ansf$\textbf{O}$rmer), which integrates occlusion-resilient 2D pose representation into transformers in a token-wise manner. Utilizing a specialized pose tokenizer, we efficiently condense 2D pose data to a compact sequence of pose tokens and feed them to the transformer together with the image tokens. This process not only ensures a rich depiction of texture from the image but also fosters a robust integration of pose and image information. Subsequently, these combined tokens are queried by vertex and joint tokens to decode 3D coordinates of mesh vertices and human joints. Facilitated by the robust pose token representation and the effective combination, we are able to produce more precise 3D coordinates, even under extreme scenarios like occlusion. Experiments on both standard and occlusion-specific benchmarks demonstrate the effectiveness of $\textbf{PostoMETRO}$, and a more than 6% performance improvement compared to our baseline model is obtained. Qualitative results further illustrate the clarity of how 2D pose can help 3D reconstruction. Code will be made available.

<div align="center">
  <img src="./assets/overview.png" alt="Overview Image">
</div>


<div align="center">
  <img src="./assets/pipeline.png" alt="Overview Image">
</div>




## Visualization 

<!-- ![Teaser 1](./assets/teaser_narrow_1.gif) ![Teaser 2](./assets/teaser_wide_1.gif) -->
We offer some GIF teasers to demonstrate the generalizability of PostoMETRO :point_down:

<div style="display: flex; align-items: center;">
  <img src="./assets/teaser_narrow_1.gif" alt="teaser 1" style="height: 300px; margin-right:20px; margin-bottom: 20px">
  <img src="./assets/teaser_wide_1.gif" alt="teaser 2" style="height: 300px; margin-right:20px; margin-bottom: 20px">
</div>

<div style="display: flex; align-items: center;">
  <img src="./assets/teaser_narrow_2.gif" alt="teaser 3" style="height: 300px; margin-right:20px;margin-bottom: 20px">
  <img src="./assets/teaser_wide_2.gif" alt="teaser 4" style="height: 300px; margin-right:20px;margin-bottom: 20px" >
</div>


We also compare our method with FastMETRO to show its improvement :eyes:

<div align="center">
  <img src="./assets/visualization.png" alt="vis">
</div>


<div align="center">
  <img src="./assets/occlusion_analysis.png" alt="occ-sens">
</div>

For more results, check our paper!

## Result

We also offer quantitive result for better comparison :point_down:

<div align="center">
  <img src="./assets/result1.png" alt="result1">
</div>
<div align="center">
  <img src="./assets/result2.png" alt="result2">
</div>
<div align="center">
  <img src="./assets/result3.png" alt="result3">
</div>
