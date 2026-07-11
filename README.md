# Under-Construction
Implementing reward functions that lead to desired behaviors in reinforcement learning is not as straightforward as it seems. Unsupervised reinforcement learning enables agents to not only learn behaviors using intrinsic rewards without extrinsic rewards, but also acquire fundamental skills that can be used to perform desired behaviors. Diversity Is All You Need (DIAYN) and Contrastive Intrinsic Control (CIC) are unsupervised methods that discover diverse behaviors from a single training process. While DIAYN distinguishes discrete skills from observations, CIC optimizes the embedding similarity between observations and continuous skills. Before going into the details, CIC was proposed to improve the behavioral diversity of DIAYN, but my implementation of CIC performed worse than DIAYN, even though I implemented the loss and intrinsic reward to match the paper as closely as possible. This may indicate that I could have missed something in the implementation or the CIC results may be sensitive to specific training conditions. Nevertheless, I chose CIC as a comparison with DIAYN because it not only provides a new perspective that diverse behaviors can be learned through continuous skill representations, but also utilizes an interesting approach with an operation similar to Transformer attention.

In my implementation, both DIAYN and CIC use SAC as the backbone because they are methods that use additional loss functions and intrinsic rewards with the backbone algorithm, and SAC is selected due to its strong exploration capability among the algorithms I have implemented so far. DIAYN is based on the paper [*DIVERSITY IS ALL YOU NEED: LEARNING SKILLS WITHOUT A REWARD FUNCTION*](https://arxiv.org/pdf/1802.06070), and CIC is based on the paper [*CIC: Contrastive Intrinsic Control for Unsupervised Skill Discovery*](https://arxiv.org/pdf/2202.00161).

## Skill, Loss and Intrinsic Reward

### Diayn
The discrete skill z is represented as a one-hot vector to treat each skill as an independent category without any numerical order. The objective is to predict the correct skill when given the next state. The next state is input to the discriminator network, which outputs logits for all possible skills. These logits are converted into probabilities using softmax, and the negative log probability of the correct skill is minimized to optimize the discriminator.


### CIC
The continuous skill z is represented as a vector containing random values between 0 and 1.

