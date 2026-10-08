---
title: "计算假设的大团块 · The Big Blob of Compute Hypothesis（双语）"
date: 2026-10-08
draft: false
visibility: public
categories: ["模型与评测"]
tags: ["AGI", "Compute", "双语原文"]
description: "The Big Blob of Compute Hypothesis 双语 EPUB 的完整 Markdown 格式转换。"
---

# 计算假设的大团块

The Big Blob of Compute Hypothesis

中英对照 · 扫描件经本地 OCR 转录


# 计算假设的大团块

# The Big Blob of Compute Hypothesis

## 一、简介 1. Introduction

我曾多次考虑写下我对通用人工智能安全性的总体看法。到目前为止我还没有这样做的主要原因是我对这个问题的看法相当高层次——它来自与保罗和埃利以泽编写的那种“框架式”议程不同的类型，事实上，其中很大一部分是认为此类框架在错误的抽象级别上运行（尽管如此，保罗的框架与我认为的东西大体兼容，而埃利泽的框架大体不兼容）。然而，后来我意识到，我的安全观点背后的核心信念也是我对 AGI 时间表的观点的核心，所以把它写下来似乎特别重要。一些警告：这不是一个精确的假设，也不是我非常有信心的假设，而且我还有额外的、更具体的理由来坚持我对 AGI 时间表和 AGI 安全性的看法。这更像是我对智能可能如何运作的高级直觉（根据经验）的升华，这使我期望事情默认按照某种方式发展，并使我对那些不面对或似乎不完全理解这一点的研究方向高度怀疑图片。我将首先从人工智能本身的角度描述这一观点，然后再讨论对安全的（微妙的）影响。

I've considered several times writing up my overall perspective on AGI safety. The main reason I haven't done so so far is that my view of the problem is pretty high-level -- it's from a different genre than the kind of "framework-y" agendas that Paul and Eliezer have written, and in fact a big part of it is arguing that such frameworks operate on the wrong level of abstraction (although that said, Paul's framework is broadly compatible with things that I think, whereas Eliezer's framework is broadly incompatible). However, then I realized that the core belief behind my safety perspective is also central to my perspective on AGI timelines, so it seemed particularly important to write it up. Some caveats: this is not a precise hypothesis or one I'm super confident in, and I have additional, more specific, reasons for holding both my views on AGI timelines and my views on AGI safety It's more like a distillation of my high-level intuitions (informed by experience) about how intelligence probably works, that causes me to expect things to go a certain way by default, and makes me a priori highly skeptical of research directions that don't confront or don't seem to fully understand this picture. I'll describe the view first in terms of just AI itself, and then later go into the (subtle) implications for safety.

看好 AGI 时间线的人和看跌 AGI 时间线的人之间的一个重大分歧是原始计算还是新算法对于 AI 进步哪个更重要。我肯定更接近“原始计算”方面，但我也认为这个问题的表述不准确——什么算作算法，原始计算到底如何应用，以及那些既不是算法也不是计算的东西（比如环境设计）呢？此外，大量计算是否仅仅足以实现 AGI（即，它是多条路径中的一条），还是实际上有某种必要”？

A big division between people who are bullish vs bearish on AGI timelines is the question of whether raw compute or new algorithms are more important to AI progress. I'm definitely closer to the "raw compute" side, but I also think the question is imprecisely stated -- what counts as an algorithm, how exactly does the raw compute have to be applied, and what about things that aren't either algorithm or compute (like design of environments)? Also, is lots of compute merely sufficient to get to AGI (i.e. it's one path among several), or is it actually somehow necessary'?

我对这种情况的描述是我所说的“计算大块”（BBOC）假设。它说，创建任何给定的智能行为主要是提供大量的、最小结构的计算能力，然后通过与丰富的环境和训练过程的交互来赋予它形状和形式，从而驱动它走向适合手头任务的行为。这些高级旋钮——训练信号（或多个信号）是什么？，代理交互的环境的性质是什么？，它接触到了多少经验？ ——构成控制智能代理执行给定任务的好坏以及具体执行方式的最有效方法。一旦设置了这些旋钮，学习执行任务主要是让代理拥有足够的计算和存储能力来学习所需的认知技能。

My picture of the situation is what I call the Big Blob of Compute (BBOC) hypothesis. It says that creating any given intelligent behavior is mostly about providing a large, minimally structured mass of computational capacity, and then giving it shape and form via interactions with a rich environment and a training process that drives it towards behavior appropriate for the task at hand. These high-level knobs -- what is the training signal (or signals)?, what is the nature of the environment the agent interacts with?, how much experience is it exposed to? -- constitute the most effective ways of controlling how well an intelligent agent performs a given task and how specifically it performs it. Once these knobs are set, then learning to perform the task is mostly about the agent having enough compute and storage capacity to learn the required cognitive skills.

一些算法设置确实很重要，但它们往往属于几个非常简单的类别——它们利用世界的简单对称性，它们有助于规范和调节计算，或者使计算图的形状更平滑且不那么尴尬。更复杂的算法思想很少有帮助，特别是所谓的“基本”

A few algorithmic settings do matter, but they tend to fall into a few very simple categories -- they exploit simple symmetries of the world, they help to regularize and condition the compute, or they make the shape of the computational graph smoother and less awkward. More complicated algorithmic ideas rarely help, and in particular supposedly "fundamental"

算法的差异（我将在下面给出示例，例如贝叶斯方法与非贝叶斯方法、基于模型的强化学习与无模型的强化学习，甚至机器学习与非机器学习方法）通常最终会重新索引或重新描述相同的底层计算，在一种空壳游戏中围绕完全相同的处理进行洗牌，同时给出表面上做一些实质性不同的事情。此外，试图严格规定计算应该做什么，或者如何将其分配到所谓的（根据人类的）任务“部分”的算法往往表现得特别差，与相同大小的非结构化计算块相比，通常会导致更糟糕的结果。更广泛地说，算法在消极意义上比积极意义上更重要——有可能提出完全阻止计算流的可怕架构，但要击败非结构化 blob 确实很难，而且也很难（也许不可能）使用算法更改来让代理做除其环境和训练过程所暗示的事情之外的任何事情。

differences in algorithms (I'll give examples below, like bayesian vs non-bayesian methods, model-based vs model-free RL, or even ML vs non-ML methods) usually end up mostly reindexing or redescribing the same underlying computations, shuffling around exactly the same processing in a kind of shell game, all while giving the surface appearance of doing something substantively different. Also, algorithms which attempt to rigidly prescribe what the compute should do, or how it should be divided up among alleged (according to humans) "pieces" of the task, tend to do particularly poorly, often leading to much worse results compared to an unstructured compute blob of the same size. More broadly, algorithms matter much more in a negative sense than a positive sense -- it's possible to come up with horrible architectures that totally block the flow of compute, but it's really hard to beat the unstructured blob by all that much, and it's also really hard (perhaps impossible) to use algorithmic changes to get the agent to do anything other than what's implied by its environment and training process.

这张图告诉了我对人工智能时间表和人工智能安全的看法。由于上面的总结有些抽象，我将首先（第 2 节）给出 BBOC 的更详细表述，并（第 3 节）给出其真实情况或我预测其将会真实的示例。然后我将（第 4 节）描述它所建议的安全观点，并（第 5 节）描述为什么这会导致对 MIRI 的 HRAD 议程的怀疑以及对“框架”式人工智能安全计划的总体怀疑。最后，我将描述（第 6 节）我认为兼容 BBOC 的安全方法应该是什么样子，以及它与安全团队目前在 OpenAI 所做的工作有何关系。

This picture informs my view of both AI timelines and AI safety. Since the summary above is somewhat abstract, I'll first give (section 2) a more detailed formulation of BBOC, and (section 3) examples of where it's been true or where I predict it's going to be true. Then I'll (section 4) describe the perspective on safety that it suggests, and (section 5) describe why that leads to skepticism of MIRI's HRAD agenda and mild skepticism of "framework"-style AI safety plans in general. Finally, I'll describe (section 6) what I think a BBOC-compatible approach to safety should look like, and how that relates to what the safety team is currently doing at OpenAI.

## 2. BBOC更详细的配方 2. More Detailed Formulation of BBOC

假设你想学习然后重复执行一些任务，比如识别图像、玩视频游戏、做科学、证明定理、模仿人类，甚至是快速学习新事物的技能。你如何做到这一点？ BBOC 假说认为，你应该主要担心以下 7 件事；其他一切通常都会产生较小的影响，或者在抽象级别太低的情况下运行而无法正确通用：

Suppose you want to learn and then repeatedly execute some task, like recognizing images, playing video games, doing science, proving theorems, imitating a human, or even the skill of learning new things quickly. How do you do this? The BBOC hypothesis says that you should worry mostly about the following 7 things; everything else usually has a minor impact or is operating at too low a level of abstraction to be properly general:

1\.

1\.

2\.

2\.

计算：您需要足够的计算来学习手头的任务。您需要计算在环境观察和代理动作之间平稳、自由地流动。它应该是相对非结构化的并且能够响应培训过程；除此之外，它的确切形式是次要的。

Compute: You need enough computation to learn the task at hand. You need the computation to flow smoothly and freely between observations of the environment and the agent's actions. It should be relatively unstructured and responsive to the training process; aside from that its exact form is secondary.

参数：代理需要一种机制来持久存储在训练期间了解的有关一般任务的信息，并暂时存储在执行期间了解的有关任务实例的信息。该信息可以是隐式的或显式的、直接存储的观察结果或统计模型的参数。 （在简单的深度神经网络中，持久和瞬态信息是权重和激活；对于进化之类的东西，它是一类生物体的基因组和特定大脑的突触权重。）如果你没有足够的参数，你将无法很好地学习任务。

Parameters: The agent needs a mechanism to persistently store information that it learns about the general task during training, and transiently store information that it learns about an instance of the task during execution. This information can be implicit or explicit, directly stored observations or the parameters of a statistical model. (In simple deep neural nets the persistent and transient information are the weights and the activations; for something like evolution it's the genome of a class of organisms and the synaptic weights of a particular brain.) If you don't have enough parameters, you won't learn the task well.

3\.

3\.

4\.

4\.

5\.

5\.

6\.

6\.

7\.

7\.

经验量：代理需要在执行任务的环境中拥有足够的经验。我使用体验这个词而不是数据，因为静态数据是与动态环境交互的特殊情况，如果代理可以在内部模拟体验的某些方面，那么体验可以在某种程度上与计算进行权衡。

Quantity of experience: The agent needs enough experience acting in the environment where the task is to be performed. I use the word experience instead of data because static data is a special case of interaction with a dynamic environment, Experience can to some extent be traded off against compute if the agent can internally simulate some aspects of experience.

经验分布：学习过程中经验的分布必须是这样的：在训练任务上表现良好的最简单方法在最终任务的实例上也表现良好。实现这一目标的一种方法是，如果训练任务实例是独立同分布的。与测试任务实例具有相同的分布，但如果训练任务以这样一种方式变化，即与单独解决每个任务相比，找到它们之间的高级共性（例如物理定律或其他一些小描述长度程序）所需的比特数更少，那么它也可以工作。那么，即使测试任务与训练任务有很大不同，只要保留高级共性，代理就会做得很好。

Distribution of experience: The distribution of experience during learning must be such that the simplest way to do well on the training tasks also does well on an instance of the final task. One way to achieve this is if training tasks instances are drawn i.i.d. from the same distribution as test task instances, but it also works if the training tasks vary in such a way that it takes less bits to find the high-level commonalities between them (say the laws of physics, or some other small description length program) than it does to separately solve each task. Then, even if the test task is very different from the training tasks, the agent will do well as long as the high-level commonalities are preserved.

除此之外，训练任务可能需要比最终任务更简单，或者难度平滑增加。

Separately from this, it may be necessary for the training tasks to be simpler than the final task, or to increase smoothly in difficulty.

归一化和调节：自由流动的数值计算很容易出现数值稳定性或调节问题。过去 5 年人工智能的许多“算法”创新实际上都是标准化和调节数值计算的简单方法：考虑 Adam、BatchNormalization 或 ResNets。

Normalization and conditioning: Freely flowing numerical computation can very easily have numerical stability or conditioning issues. Many of the "algorithmic" innovations in the last 5 years of AI have actually been simple methods for normalizing and conditioning numerical computation: consider Adam, BatchNormalization, or ResNets.

这些方法基本上只是说，在计算的某些方面出现的数值量应该归一化为 1 或重新参数化为与 1 的差异。这种简单的“流程控制”通常胜过深度学习内部或外部的更复杂的想法。

These methods basically just say that numerical quantities that occur in some aspect of the computation should be normalized to 1 or reparameterized as a diff from 1. This simple "flow control" has often outperformed much more sophisticated ideas either within or outside deep learning.

对称性和形状：最具影响力的算法创新的另一个重要部分是使用对称性来减少模型中的参数数量，从而使模型学习得更快。智能之所以起作用，是因为我们生活在一个低熵世界；低熵的体现之一是物理世界的对称性和稀疏性，例如，卷积网络利用了空间平移不变性； RNN 利用时间平移不变性（内存架构可能是未来的候选者，因为它们利用一般稀疏性），我们可能可以在没有这些对称性的情况下学习相同的任务，但这将需要更多的计算和参数（最终会自行发现对称性）。同样，计算的形式也很重要，我们不希望信息在非常小的层中成为瓶颈，或者在分析观测值之前被迫压缩它们。计算的形式应该灵活且合理。随着任务变得更加普遍，对称性可能变得不那么重要，因为它们可能不适用于跨任务，并且我们还可以训练元学习架构，在对象级别上试验各种对称性。

Symmetries and shape: Another huge chunk of the most impactful algorithmic innovations has been the use of symmetry to decrease the number of parameters in models, and thus make them learn much faster. Intelligence only works at all because we live in a low entropy world; one way the low entropy manifests is symmetries and sparsity in the physical world, For example, convolutional nets take advantage of spatial translation invariance; RNNs take advantage of time translation invariance (memory architectures might be a future candidate in that they take advantage of general sparsity), We could likely learn the same tasks without these symmetries, but it would take a lot more compute and parameters (which would then eventually discover the symmetries on their own). In a similar vein, the shape of the computation can matter we don't want information to bottleneck at a very small layer, or be forced to compress observations before analyzing them, The shape of the computation should be flexible and sensible. As tasks get more general, symmetries may become less important, as they may not hold across tasks and we may also be able to train meta-learning architectures that experiment with various symmetries at the object level.

训练目标：计算需要由当前任务的正确评估机制驱动，该机制可以是损失函数、奖励、预测误差、GAN 损失、适应度函数、人类评估、扩展的人类交互或组合上述任何一项的过程。您将学到训练过程渐进激励的任何内容；如果你的训练过程错误，你就会学到错误的任务。这不仅仅是对安全性的评论：今天的许多深度学习论文都学习了错误的任务：例如，试图在对话中匹配人类响应的对话系统（人类对话的目标不是匹配其他人的响应，而是完成人类在对话中的任何目标），非元学习系统在一项任务上进行训练，然后将结果应用到另一项任务上（代理没有接受过转移到另一项任务的训练，因此它在这方面不是最佳的）。请注意，“训练目标”不一定是单个目标函数或奖励函数；它可能是一个复杂的过程，由多个部分共同推动达到某种渐近条件。例如，强化学习中来自人类反馈的训练目标来自于两种不同学习过程的结合，类似于“学习在环境中采取行动，使人类更喜欢你的动作短片，而不是你可以采取的其他动作的短片”。

Training target: The computation needs to be driven by the correct evaluative mechanism for the task at hand, which can be a loss function, a reward, a prediction error, a GAN loss, a fitness function, a human evaluation, an extended human interaction, or a process for combining any of the above. You will learn whatever the training process asymptotically incentivizes; if you have the wrong training process, you'll learn the wrong task. This isn't just a comment on safety: many deep learning papers today learn the wrong task: for example, dialog systems that attempt to match human responses in a conversation (the goal of human conversation isn't to match the responses of some other human, but to accomplish whatever the human's goal is in the conversation), non-metalearning systems that train on one task and then apply the results to another (the agent was not trained on the task of transferring to another task, therefore it won't be optimal at that). Note that the "training target" is not necessarily a single objective function or reward function; it can be a complicated process with several parts that together drive towards some asymptotic condition. For example, the training target in RL from human feedback comes from combining two different learning processes and is something like "learn to act in the environment in such a way that short clips of your actions would be preferred by a human relative to short clips of other actions you could take".

BBOC 表示，“如果你答对了 1-6，你可能会学到 7 指定的行为（如果它可以通过计算上易于处理的方式学习）。未包含在 1-7 中的事物重要的可能性要低得多，尽管也有例外。”

BBOC says something like "if you get 1-6 right, you will probably learn the behavior that 7 specifies if it's learnable in a computationally tractable way. Things not included in 1-7 have a much lower likelihood of being important, although there are exceptions."

与 BBOC 相关的概念（可能有助于直觉）是我所说的智力“雪花模型”。如果你看过雪花的分形结构，你可能会认为无论是谁制造了它，都做了一些不可能的复杂和困难的事情，但是一点一点地建造它一定是可能的，因为有人做到了。事实上，这两种说法都是错误的：制造雪花的方法不是从它的碎片角度来思考，而是要了解物理定律（训练目标），有足够的原材料（计算）和足够大的房间（参数），正确设置温度、压力和湿度（标准化、形状），并等待足够长的时间（经验量）。此外，这是制作雪花的唯一方法，也是影响雪花形状的唯一手段；试图将一小块冰块拼凑成一个，基本上是没有希望的。对于 snoMlakes，您可以将其称为“大雪团”理论。

A related concept to BBOC (that may help with intuition) is what I call the "snowflake model" of intelligence. If you looked at the fractal structure of a snowflake, you might think that whoever made it did something impossibly intricate and difficult, but that building it piece by piece must somehow be possible because someone did it, In fact, both statements are false: the way to make a snowflake is not to think in terms of its pieces but to know the laws of physics (training target), have enough raw material (compute) and a large enough chamber (parameters), set the temperature, pressure, and humidity correctly (normalization, shape), and wait for long enough (quantity of experience). Furthermore, this is your only way to make snowflakes and your only leverage over their shape; trying to piece together a single one from little bits of ice is basically hopeless. For snoMlakes you might call this the "big blob of snow" theory.

Ilya Sutskever 将 BBOC 表示为“网络想要学习”。

Ilya Sutskever expresses BBOC as "networks want to learn".

值得注意的是一些例外或明显的例外。当训练某个任务的狭窄版本时（比如玩单个 atari 游戏，而不是玩任何视频游戏），通常可以专门针对该任务手工设计一种方法，从而带来明显的大规模算法改进和更规定的计算结构，但这是以脆弱性和对更广泛任务的泛化性较差为代价的（例如，考虑大量提出对 atari 上的 RL 进行小调整的论文）。这种（通常在不知不觉中）过度适应狭窄任务的现象造成了算法创新成功的假象，同时也使我们的智能体显得脆弱且过于复杂，并掩盖了计算和结果之间的简单关系。我认为，对于那些主要关注小任务并期望突破首先出现的人来说，这会给这个领域带来非常扭曲的看法。相比之下，大型项目的经验似乎可以培养 BBOC 直觉，而且我也相信，当我们转向更通用的任务时，BBOC 将变得更加明显（这在算法上比更狭窄的任务更容易学习）。

It's worth noting some exceptions or apparent exceptions. When training on a narrow version of a task (like playing a single atari game, rather than playing any video game), it's often possible to hand-engineer a method for that task in particular, leading to apparent large algorithmic improvements and more prescribed structure to the compute, but this comes at the cost of brittleness and poor generalization to broader tasks (e.g. consider the flood of papers proposing small tweaks to RL on atari). This phenomenon of (often unknowingly) overfitting to narrow tasks creates the illusion of successful algorithmic innovation, while also making our agents appear brittle and overcomplicated, and masking the simple relationship between compute and results. I believe this gives a very distorted view of the field for people who mainly pay attention to small tasks and expect breakthroughs to appear there first. By contrast, experience with huge projects seems to cultivate the BBOC intuitioni, and I also believe that BBOC will become more obvious as we move to more general tasks (which will be algorithmically simpler to learn than more narrow tasks).

上面段落的一个极端情况是任务范围如此狭窄，以至于您根本不需要任何参数，人们可以简单地编写一个简短的程序来解决它们（或尝试一小部分此类程序）。此类示例可能是排序、图形导航或 alpha-beta 修剪。这里，该方法很好地解决了狭窄的任务，根据您使用的算法，效率可能会更高或更低，并且在该狭窄任务的实例内完美地概括（例如，对较大的数字列表进行排序），但在该任务之外的概括能力为零（即，当您给它一个除排序之外的算法任务时，它会很好地响应）。基本上，在机器学习之前出现的所有人工智能工作都是这种形式——机器学习之前的人工智能大多只是专注于某些选定领域的编程。这段历史掩盖了 BBOC，并使它看起来像是可以从算法中获得巨大收益，并且正确的算法可以提供完美的泛化。然而，实际上，泛化超出这些单一算法任务需要搜索程序，这将不可避免地是启发式的，将不可避免地拖累学习/机器学习/参数，并且如果任务足够通用（例如解决任意编程面试问题）将需要大量计算并且将遵循 BBOC 的正常规则。

An extreme case of the paragraph above is the case of tasks that are so narrow that you don't need any parameters at all and a human can simply write a short program that solves them (or experiment with a small range of such programs). Examples of this might be sorting, graph navigation, or alpha-beta pruning. Here the method solves the narrow task really well, can be much more or less efficient depending on the algorithm you use, and generalizes perfectly within instances of that narrow task (for example, sorting a larger list of numbers), but has zero ability to generalize outside that task (i.e. responding well when you give it an algorithmic task other than sorting). Basically all AI work that came before ML was of this form -- pre-ML AI was mostly just programming focused on certain selected domains. This history obscures BBOC and makes it look like huge gains from algorithms are possible and that the right algorithms give perfect generalization. In reality, however, generalizing just a bit beyond these single algorithmic tasks requires searching over programs, which will inevitably be heuristic, will inevitably drag in learning/ML/parameters, and if the task is general enough (e.g. solve an arbitrary programming interview question) will require a lot of compute and will follow the normal rules of BBOC.

正如所写的，BBOCt 的大部分描述读起来可能像是设计深度学习系统的常识指南，但大多数人并没有仔细考虑它延伸到什么程度或者它对智能意味着什么。人们通常所说的深度学习的许多内容，无论是在该领域内还是在该领域外，都与 BBOC 强烈矛盾，如果你真的相信 BBOC，那么听起来就很愚蠢。下面我举了一些例子，说明人工智能训练中什么重要、什么不重要，这有助于表明它是多么令人惊讶和深远。

As written, much of BBOCts description may read like a common sense guide to designing deep learning systems, but most people have not thought carefully about how far this extends or what it implies about intelligence. Many things that people commonly say about deep learning, both within the field and outside it, are strongly contradicted by BBOC, and start to sound silly if you really believe BBOC. Below I give some examples about what matters and what doesn't in AI training, that helps to make clear how surprising and far-reaching it is.

## 3. BBOC的证据和例子 3. Evidence and Examples of BBOC

以下是我在 3 年训练 AI 系统和多年研究大脑过程中观察到的一些事情，它们要么是 BBOC 的证据，要么是 BBOC 的影响/预测：

Here are some things I've observed over 3 years of training AI systems and many years of studying the brain, that are either evidence for BBOC, or implications/predictions of BBOC:

• BBOC 的经典（也是最著名）示例是使用卷积神经网络进行视觉。在 2012 年之前的几年里，人们精心尝试设计边缘检测器、轮廓集成系统、分割系统、前景-背景分析器、姿态估计器等，然后人们试图将它们组合在一起制作视觉系统。这些系统失败是因为它们没有使用足够的计算（1），没有足够的参数（2），组织计算过于严格（6），并且没有正确的目标函数（这些部分是在边缘检测等方面进行训练的，而不是在

• The classic (and best known) example of BBOC is the use of convolutional neural nets for vision. In the years before 2012 there were elaborate attempts to design edge detectors, contour integration systems, segmentation systems, foreground-background analyzers, pose estimators, and so on, which people then tried to put together to make vision systems. These systems failed because they did not use enough compute (1), did not have enough parameters (2), organized the computation too rigidly (6), and did not have the right objective function (the pieces were trained on e.g. edge detection rather

我在人工智能领域的第一份工作是研究 Deep Speech 2，我现在认为这是当时训练过程中计算使用量最大的项目，它在 2015 年每个训练模型使用了约 0.3 petaflop 天。

My first job in AI was working on what turned out to be what I now believe was the single largest use of compute in one training process up to that time, Deep Speech 2, which used about 0.3 petaflop-days per trained model in 2015.

• 比最终对象的识别）(7)。 CNN 解决了​​所有这些问题。如果你保留复杂的系统，但修复（1）、（2）和（7），它们在 imagenet 上可以做得还不错，但它们仍然效率低下、复杂，并且泛化性很差。

• than identification of the final object) (7). CNN's fixed all these problems. If you keep the elaborate systems but fix (1), (2), and (7), they can do passably on imagenet, but they are still inefficient, complicated, and generalize poorly.

基本上，我训练 AI 系统的全部经验都与 CNN 案例一致——尽可能使用最简单的架构和算法，并正确完成 (1)-(7)，你就会做得很好。如果您没有足够的计算、参数或数据，那么没有什么可以拯救您。

Basically my entire experience with training AI systems lines up with the CNN case -- use the simplest architecture and algorithm you can, and get (1)-(7) right, and you'll do well. If you don't have enough compute, parameters, or data, nothing can save you.

语音、语言建模、翻译、强化学习、生成模型，以及很快的机器人技术——它们似乎都遵循这种模式。只需几句话就可以说明这一事实，但是经验的重要性，训练数百个模型，尝试数百件事，看看什么有效，什么无效，是很难夸大的。

Speech, language modeling, translation, reinforcement learning, generative models, soon robotics -- they all seem to follow this pattern. It takes only a few words to state this fact, but the weight of experience, of training hundreds of models, trying hundreds of things, and seeing what works and what doesn't, is hard to overstate.

当我在谷歌时，我参与了一个项目，看看神经网络对于深度学习是否真的至关重要。我们尝试用某种其他类型的机器学习模型（我不允许具体说出什么）来替换每层中的矩阵&gt;（+非线性），基本上是一种“深度&lt;其他一些机器学习技术&gt;”。我们发现这些模型可以工作，但始终表现得比类似大小的神经网络稍差。最终我发现，如果愿意的话，这些自定义层中的任何一个都可以以其参数的子集形成一个标准的神经网络层，这就是它选择做的事情；基本上它已经重新参数化了我们的模型换句话说，不同的层实际上导致了神经网络本来可以完成的相同计算的重新排列，一方面，这告诉我们深度神经网络是非常有效的模型（其他模型也想成为它们），另一方面，它告诉我们其他模型也可以工作，并且“深度学习”没有什么神奇之处；更重要的是，某些计算确实想要发生，并且会围绕您提供的任何类型的模型进行塑造。

When I was at Google I was involved in a project to see if neural nets are really essential to deep learning. We tried to replace the matri&gt;(+nonlinearity in each layer with some other kind of machine learning model (l am not allowed to say exactly what), basically making a "deep &lt;some other ML technique&gt;". We found that these models worked but consistently performed slightly worse than similarly-sized neural nets. Eventually I discovered that any of these custom layers could form a standard neural-net layer with a subset of its parameters if it wanted to, and that is what it had chosen to do; basically it had reparameterized the model we gave it to internally and surreptitiously form a neural net. In other words, the different layer literally caused rearrangement of the same computation that a neural net would have done. On one hand, this tells us that deep neural nets are pretty effective models (other models want to become them), on the other hand, it tells us that other models can work too and there's nothing magic about "deep learning"; it's more that certain computations really want to occur and will shape themselves around whatever type of model you give them.

在强化学习教科书中，基于模型的强化学习和无模型强化学习之间存在区别。第一个有一个可以进行预测的环境模型和一个策略，第二个只有一个策略。理论上，基于模型的强化学习有点像系统 2，因为它可以提前思考和计划，而无模型强化学习则类似于系统 1。在实践中，无模型强化学习倾向于在内部进行一些规划，我们开始怀疑，如果我们只是以不同的方式制定无模型策略，比如允许它们在采取行动之前迭代许多步骤，那么它们将能够（隐式）至少与基于模型的系统一样好地进行规划。与此同时，实际的基于模型的强化学习系统的性能相对较差，这可能是因为我们人为地强制计算在两个不相连的块中进行。关键是，规划的计算活动很重要，但该活动是否发生更多地与计算块的形状有关，而不是与我们假设使用的算法或哪个神经网络应该做什么有关。当环境和目标函数需要时，“无模型”系统会在内部学习模型，如果有时是一个有缺陷的模型，则可能只是因为我们不倾向于塑造无模型网络，以便为它们提供足够的内部递归来隐式进行类似模型的计算。

In RL textbooks there is a distinction between model-based RL and model-free RL. The first has an environment model that makes predictions and also a policy, the second has just a policy. In theory model-based RL is a bit like system 2 in that it can think ahead and plan whereas model-free RL is like system 1. In practice model-free RL tends to do some planning internally, and we're starting to suspect that if we simply shaped model-free policies differently, say allowing them to iterate themselves for many steps before taking an action, that they would be able to (implicitly) do planning at least as well as model-based systems. Meanwhile actual model-based RL systems perform relatively poorly, perhaps because we're artificially forcing computation to take place in 2 disconnected blocks. The point is, the computational activity of planning is important, but whether or not that activity occurs has more to do with the shape of the blob of computation than with the algorithm we're supposedly using or which neural net is supposed to do what. "Model-free" systems learn a model internally when their environment and objective function require them to, and if it's sometimes a deficient model that may just be because we don't tend to shape model-free networks so as to give them enough internal recurrence to implicitly do model-like computations.

同样，普通的 RL 据说也存在长期问题，例如规划超过 100 万个时间步长。分层强化学习应该可以解决这个问题；这个想法是有目标和子目标，不同的政策在不同的时间尺度设定目标。不幸的是，到目前为止，分层强化学习还无法比在普通强化学习算法中隐式学习的目标和子目标更好地设置目标和子目标。层次结构有一种倾向，就是崩溃成单个（更深的）神经网络所做的事情。我们最终可能会让分层强化学习发挥作用，但目前看来，这似乎是计算的另一个例子，它对自己想要做什么有自己的想法，并围绕由一个所谓的重要算法强加给它的结构进行路由。

Similarly, ordinary RL is said to have issues with long time horizons, say planning over 1 million timesteps. Hierarchical RL is supposed to deal with this; the idea is to have goals and subgoals, and different policies set the goals at different timescales. Unfortunately so far hierarchical RL hasn't been able to set goals and subgoals much better than they can be learned implicitly within a normal RL algorithm; there's a tendency for the hierarchy to just collapse into approximately what a single (deeper) neural-net would have done. It's possible we'll eventually make hierarchical RL work, but for now it seems to be yet another example of compute having its own idea about what it wants to do and routing around the structure imposed on it by a supposedly important algorithm.

与此同时，当我们将策略设置得非常大时，普通的 RL 算法在长期范围内取得了令人惊讶的成功，有时每 10,000 个时间步长只需要 1 个奖励来解决任务。我们最终确实需要在 RL 策略中表示层次结构，但这可能会隐式发生，而不是通过我们标记为“层次 RL”的技术。 LSTM 据说比简单的 RNN 更好，因为它们可以更长时间地存储状态，这是深度学习中为数不多的真正的算法创新之一。然而，现在有证据表明，如果调整得当，简单的 RNN 的性能与 LSTM 一样好。 LSTM 的唯一优点是它们适用于更广泛的超参数。事实上，在保持参数数量和计算量不变的情况下，不同的循环架构（RNN、LSTM、GRU）似乎具有惊人的相似容量和性能。当当前的神经网络泛化能力不佳时，人们经常谈论这一点，就好像这是某种算法缺陷，就好像在某个地方有一些尚未发现的算法可以更好地泛化2。但我的经验是，当你将神经网络暴露在更广泛的数据分布中，迫使它们找到通用解决方案而不是仅仅记住一些特殊情况时，神经网络就能更好地概括。例如，在训练语音模型时，我发现如果你只训练美国口音，那么你在许多其他口音上都会表现得很差。然而，如果你训练 5 或 6 种口音，你会立即在看不见的口音上表现出色。可能发生的情况是，对于 1 或 2 种口音，记住每种口音的特殊性很容易，而随着数量的增加，识别它们之间的共性并存储它们变得更容易（并且需要更少的位），从而导致泛化。总结一下：导致泛化的不是算法，而是训练设置和环境。我有时会听到学术 ML 领域和 AI 安全领域的讨论，即深度神经网络缺少某些东西，而贝叶斯方法在某些方面会做得更好（它们代表不确定性，可以更好地校准，我们更容易理解他们的信念，等等）。但事实上，神经网络已经代表了不确定性（例如类概率），如果愿意的话，甚至可以输出完整的联合分布而不是因子概率，就像理想化的贝叶斯方法一样。问题是，在神经网络和某些理想化贝叶斯模型的情况下，这将需要指数级更多的参数；两者都遇到相同的实际限制。

Meanwhile, ordinary RL algorithms are having surprising success on long time horizons when we make the policies really large, sometimes solving tasks with only 1 reward every 10,000 timesteps. We really will eventually need to represent hierarchical structure within our RL policies, but that may happen implicitly rather than through the techniques we've labeled "hierarchical RL". LSTM's are supposedly better than simple RNN's because they can store state for longer one of the few real algorithmic innovations in deep learning. Yet there's now evidence that simple RNN's perform as well as LSTMs when properly tuned; the only advantage of LSTMs is that they work well for a wider range of hyperparameters. In fact, holding constant the number of parameters and the amount of compute, different recurrent architectures (RNN, LSTM, GRU) seem to have eerily similar capacity and performance, When current neural nets don't generalize well people often talk about this as if it's some kind of algorithmic defect, as if there's some yet-to-be-discovered algorithm somewhere that generalizes drastically better2. But my experience is that neural nets generalize better when you simply expose them to a wider distribution of data that forces them to find a general solution rather than just memorizing a few special cases. For example, when training speech models I found that if you trained only on an American accent you would do poorly on many other accents. However if you train on 5 or 6 accents you will immediately do well on an unseen accent. What's probably going on is that for 1 or 2 accents it's easy to just memorize the peculiarities of each one, whereas as the number increases as it becomes easier (and requires fewer bits) to identify the commonalities between them and store these, which leads to generalization. To summarize: it's not algorithms that lead to generalization, so much as the training setup and environment. I sometimes hear discussion, in both the academic ML world and in the AI safety world, that deep neural nets are missing something and that Bayesian methods would do better in some way (they'd represent uncertainty, be better calibrated, it would be easier for us to understand their beliefs, etc). But in fact neural nets already do represent uncertainty (with e.g. class probabilities), and could even output a full joint distribution rather than factored probabilities if they wanted to, just like idealized Bayesian methods. The problem is this would take exponentially more parameters, in both the case of neural nets and some idealized Bayesian model; both run into the same practical limitation.

人们还说，如果你有一个贝叶斯模型，你就会明确地理解它相信什么命题以及相信多少，但这对我来说从来没有意义——困难

People also say that if you had a Bayesian model, you'd understand explicitly what propositions it believes and how much, but this has never made sense to me -- the hard

2 \|我相信这来自于机器学习之前的人工智能系统的经验，选择正确的算法意味着你可以在单个算法任务中完美地概括，基本上是因为这种“人工智能”只是编程。我在上一节中对此进行了一些讨论。

2 \| believe this comes from the experience with pre-ML AI systems, where picking the right algorithm meant you could generalize perfectly within a single algorithmic task, basically because this kind of "AI" was just programming. I discuss this a bit in the previous section.

部分是弄清楚如何将世界分割成“语句”，这必须通过您用来保存和填充模型的任何结构（无论是否是贝叶斯结构）来学习（例如，贝叶斯网络或神经网络）。一旦你做到了这一点，贝叶斯网络节点中持有的信念就可能是可理解或不可理解的、校准或未校准的，就像神经网络内部的激活一样，后者似乎对全世界来说都持有信念并权衡证据，即使它们没有明确标记为这样做。最后，人们还说贝叶斯模型更擅长泛化，避免过度拟合。在理想限制下这是正确的（因为您正在对分布进行建模），但在实践中，深度学习为处理过度拟合而临时提出的最简单的临时方法（例如 dropout）可以被证明与贝叶斯近似等效。与此同时，人们实际上已经尝试过明确的贝叶斯神经网络（权重具有不确定性），但它们在泛化或校准方面并没有表现出明显的优越性；正如您所期望的，它们的行为类似于 dropout 等方法。这里的关键点再次是：算法本身并不像它们看起来那么重要，并且给人一种不同的感觉，而实际上它们通常只是重新标记相同的计算过程。

part is figuring out how to carve the world up into "statements", and this has to be learned by whatever structure, Bayesian or not, you use to hold and fill your model (for example, a Bayesian network or a neural net). Once you've done that, the beliefs held in the nodes of a Bayesian network are just as likely to be comprehensible or incomprehensible, calibrated or uncalibrated, as the activations inside a neural network, the latter of which appear for all the world to be holding beliefs and weighing evidence even though they're not explicitly labeled as doing so. Finally, people also say that Bayesian models are better at generalization and avoiding overfitting. This is true in the ideal limit (since you're modeling a distribution), but in practice the simplest ad hoc methods that deep learning has improvised to deal with overfitting, like dropout, can be shown to be equivalent, to a Bayesian approximation. Meanwhile, people have actually tried explicitly Bayesian neural nets (with uncertainty on the weights), but they don't stand out as as clearly superior in generalization or calibration; they behave similarly to methods like dropout, as you'd expect. Once again the key point here is: the algorithms per se matter less than they appear, and give the appearance of something different going on when in fact they often just relabel the same computational process.

人们似乎对对抗性例子感到非常惊讶——他们说这意味着神经网络并没有真正理解他们正在分类的对象，或者在某种程度上是脆弱的。但 BBOC 对此并不感到惊讶：分类器并未接受过包含以这种方式设计的图像的经验分布的训练。根据 BBOC 的说法，当前对抗性样本的最佳防御方法是直接对其进行训练。当前第二好的防御（我认为最终将成为解决方案）是拥有一个生成模型来告诉您什么是自然图像或不是自然图像；这是有效的，因为生成模型是使用目标函数进行训练的，该目标函数告诉它区分数据流形上的内容与非数据流形上的内容，而分类器没有这样的目标函数。 BBOC 说你得到你优化的东西；你无法得到的是人类对系统应该具有哪些属性的想法。

People seem very surprised about adversarial examples -- they say it means neural nets don't really understand the objects they're classifying, or are brittle in some way. But BBOC isn't surprised by this: the classifiers weren't trained on a distribution of experience that included images engineered in this way. In line with BBOC, the current best defense against adversarial examples is to train on them directly. The current second best defense (and the thing I think will ultimately be the solution) is to have a generative model that tells you what is or isn't a natural image; this works because the generative model was trained with an objective function that tells it to distinguish what is on the data manifold versus what isn't, whereas the classifier had no such objective function. BBOC says you get what you optimize for; what you don't get is a human's idea of what properties the system should have.

新闻文章偶尔会谈论我们如何需要一些反向传播的替代方案，就好像我们缺少一些关键的算法见解，某种新的更新规则，它将彻底改变我们所做的一切，并允许我们的网络最终以真正像人类一样的方式学习（经常提到“赫布学习”）。我们很有可能找到一个新的更新规则（事实上，我怀疑许多类似梯度的更新会相对较好），但我不认为任何此类更改只会产生非常有限的效果；在高维空间中导航是很困难的，根据经验，我们尝试了很多方法，但最终只比简单梯度稍微好一点。

News articles occasionally talk about how we need some alternative to backprop, as if we're missing some key algorithmic insight, some new kind of update rule that will revolutionize everything we do and allow our networks to finally learn in a truly human-like way ("hebbian learning" is often mentioned). It's quite possible we could find a new update rule (in fact I suspect many gradient-like updates would be comparably good) but I don't expect any such change to have anything more than a very modest effect; navigating high-dimensional spaces is hard and empirically we've tried a huge number of things that end up doing only modestly better than the simple gradient.

将这种近乎图腾的重要性归因于更新规则是 BBOC 旨在抵消的一种思维的一个例子。

Ascribing this almost totemic importance to the update rule is an example of the kind of thinking BBOC is designed to counteract.

有一段时间，人们通过在一堆任务上训练策略，然后尝试对新任务进行微调来进行迁移学习。人们尝试了一系列复杂的变体，但效果都同样糟糕。这实际上是错误的训练程序（它不会渐近地导致快速适应的策略），当人们切换到正确的目标函数时，事情开始变得更好一些：训练一个在快速适应后针对其性能进行优化的初始策略（意思是，对快速适应新环境的任务进行学习和评估）。同样，详细的算法并没有多大区别；似乎有帮助的是拥有正确的目标。

For a while people were doing transfer learning by training a policy on a bunch of tasks and then trying to fine tune on a new task; people tried a bunch of complicated variants of this that all worked about equally poorly. This is actually the wrong training procedure (it doesn't asymptotically lead to fast-adapting policies) and things started working somewhat better when people switched to the right objective function: training an initial policy optimized for its performance after fast adaptation (meaning, learning and evaluation on the task of adapting quickly to a new environment). Again, the detailed algorithm didn't make much difference; what seems to help is having the right target.

有很多算法试图更好地探索强化学习，或者更好的内在动机，或者尝试更安全地探索等等。很多聪明人（包括我）都在研究这些问题，并提出了似乎得到改进的算法，但随着时间的推移，这些算法已经变成了混乱且笨重的拼凑物，有点让人想起 CNN 之前的边缘检测器。人们很容易认为我们已经超越了这一点，因为这些方法使用端到端的深度强化学习，但也许我们需要提升一个抽象层次，学习许多代理学习过程如何探索、如何激励以及如何安全地探索。

There are a lot of algorithms out there that attempt to get better exploration for reinforcement learning, or better intrinsic motivation, or that attempt to explore more safely, etc. A lot of smart people (including me) have worked on these problems, and come up with algorithms that seem to get improvements, but over time these have become a messy and unwieldy patchwork, somewhat reminiscent of the pre-CNN edge detectors. It's easy to think we're beyond that because these methods use end-to-end deep RL, but maybe we need to go up one level of abstraction, learning over many agent learning processes how to explore, how to motivate, and how to explore safely.

我们已经看到这种类型的“元学习”的早期进展，但最终的发展方向还没有定论。

We've seen early progress on this type of "metalearning", but the jury is still out on where things will ultimately go.

那么非机器学习系统（例如搜索或推理系统）又如何呢？难道他们不是在做一些真正不同的事情吗？你难道不希望他们能够更好地概括吗，因为它们包含一个符号配方，而机器学习系统却没有？我认为这里发生的是一个误解——机器学习系统完全有能力驱动符号计算，正如神经网络驱动定理证明 S.hQ& 中的所有工作一样，它们还可以根据算法或程序如何拟合数据来搜索（您可以使用神经网络来进化代码，尽管还不是很好）。事实上，我认为，如果神经网络系统的形状和结构以正确的方式组织，那么暴露于只有高级相似性（例如物理定律）的各种环境中的神经网络系统将不可避免地收敛于实现简单程序的参数（例如，它们将导出或代表物理定律），尽管我们的大脑基于非常模糊的计算，但还是做到了这一点。一旦发生这种情况，神经网络将搜索程序，而非机器学习方法代表固定程序（或小系列程序，如树搜索）；那时，非 ML 方法将仅执行 ML 方法3 完成的计算的固定、严格的子集。因此，我们可以预期，ML 系统将比非 ML 系统做得更好（即使是在符号任务上），同时使用更多的计算来实现这一点，这与 BBOC 一致。我们已经可以通过 MCTS 看到这一点，DeepMind 发表了一些论文，表明学习搜索可以胜过固定树搜索算法，并且将其纳入例如搜索算法中只是时间问题。阿尔法狗。

What about non-ML systems, like search or reasoning systems? Aren't they doing something genuinely different, and wouldn't you expect them to generalize better because they contain a symbolic recipe, while ML systems don't? What I believe is going on here is a misunderstanding -- ML systems are perfectly capable of driving symbolic computation, as all the work in neural-net driven theorem proving S.hQ&, and they can also search over algorithms or programs based on how they fit the data (you can use neural nets to evolve code, though not well yet). In fact I think neural net systems exposed to a wide variety of environments with only high-level similarities (e.g. the laws of physics) will, if their shape and structure is organized in the right way, inevitably converge on parameters that implement simple programs (e.g. they will derive or represent the laws of physics) our brains did this despite being based on very fuzzy computation. Once that happens, neural nets will be searching over programs whereas non-ML methods represent fixed programs (or small families of programs, like tree search); at that point non-ML methods will just be doing a fixed, rigid subset of the computation done by ML methods3. So we can expect that ML systems will do much better than non-ML systems (even at symbolic tasks) while using more computation to do so, in line with BBOC. We can already see this with MCTS DeepMind has published some papers showing that learning to search can outperform fixed tree search algorithms, and it's only a matter of time until this is incorporated into e.g. AlphaGo.

考虑人类的进化。大自然是如何发现并发展出能够揭示物理定律、证明费马大定理以及创作交响乐的大脑的？你可以一步步追踪它是如何发生的，并对每一步给出解释，但整个过程的力量仍然令人惊讶。 BBOC 提供了一个简单的高级解释，足够的计算、足够的最低结构大脑记忆、足够广泛的环境经验和挑战分布，以及涉及让代理相互对抗的驱动目标，足以使结果最终不可避免。

Consider human evolution. How did nature manage to discover and develop brains capable of uncovering the laws of physics, proving Fermat's last theorem, and composing symphonies? You can trace step by step how it happened, and give explanations for each step, yet the process as a whole is still surprising in its power. BBOC offers a simple high-level explanation enough compute, enough minimally structured brain memory, a broad enough distribution of environment experience and challenges, and a driving objective that involved playing agents off against each other were sufficient to make the result inevitable eventually.

3 上一节也对此进行了讨论：早期的人工智能工作侧重于解决特定狭窄问题的固定程序或小型程序系列；将这项工作称为“AI”会掩盖 BBOC，并错误地让我们认为符号计算有什么不同。

3 This is also discussed in the section above: early AI work focused on fixed programs or small families of programs that solve specific narrow problems; calling this work "AI" obscures BBOC and erroneously leads us to think there's something different about symbolic computation.

希望上面的内容能够解释为什么我认为 BBOC 如此有可能，尽管乍一看似乎很奇怪，并且从小规模基准来看并不明显。

Hopefully the above gives some flavor of why I see BBOC as so likely, despite seeming strange at first glance and not being obvious from looking at small-scale benchmarks.

请注意，这并不意味着扩大当今的精确算法将导致通用人工智能。它确实表明，任何算法的改变都可能是利用对称性或稀疏性的简单事物（好的候选者可能是基于记忆的模型，或在训练过程中尺寸不断增长的神经网络），或训练过程中的创新，而规模将是故事的重要组成部分。它还表明，“深度学习的替代品”要么不起作用，要么只是重述深度学习，因此面临很高的风险，并且应该对据说引入一些理想特性的算法创新持怀疑态度。

Note that none of this implies that scaling up today's exact algorithms will lead to AGI. It does suggest that any algorithmic changes will likely be simple things that exploit symmetries or sparsity (good candidates might be memory-based models, or neural nets that grow in size over the course of training), or innovations in the training process, and that scale will be a huge part of the story. It also suggests that "alternatives to deep learning" are at very high risk of either not working or just recapitulating deep learning, and that algorithmic innovations said to introduce some desirable property should be treated with skepticism.

BBOC 的一个显着特征是，一个研究方向可能在很长一段时间内看起来完全合理（视觉边缘检测器、基于模型的强化学习、分层强化学习、探索），并且有许多最聪明的人在研究它，然后才清楚地表明它只是重新安排计算或试图对如何解决问题过于规范。然而，随着时间的推移，您会对某种方法“过于手工制作”产生一种直觉，类似于程序员对抽象不够通用或模块化的直觉。基于模型的、分层的和探索仍然存在争议——也许我们仍然会从中得到一些东西——但它们开始看起来越来越可疑，越来越像 2011 年的边缘检测器。

A striking feature of BBOC is that a research direction may look perfectly reasonable for a long time (visual edge detectors, model-based RL, hierarchical RL, exploration) and have many of the smartest people working on it, before it becomes clear that it's simply rearranging compute or trying to be too prescriptive about how to solve a problem. Over time, however, you develop an intuition for when an approach is "too handcrafted", similar to a programmer's intuition about when an abstraction is insufficiently general or modular. Model-based, hierarchical, and exploration are still controversial -- maybe we'll still get something out of them -- but they are starting to look increasingly suspicious, increasingly like the edge detectors of 2011.

最后要注意的是，有时很难或不可能立即采用 BBOC 推荐的方法来解决问题。有时，您需要从较小问题的较低级别开始（例如，在一款视频游戏上从普通深度强化学习开始，然后再在许多视频游戏上尝试元强化学习），以了解方向并了解如何设计更高级别的算法。因此，设计较低级别架构的较小项目可以作为垫脚石，当我讨论我的人工智能安全研究策略（第 6 节）时，我会再次讨论这一点。

A final note is that sometimes it's hard or impossible to take the BBOC-recommended approach to a problem right away. Sometimes you need to start with something lower-level on a narrower problem (e.g. start with ordinary deep RL on one video game before you try meta-RL on many video games) to get your bearings and understand how to design the higher-level algorithm. So narrower projects where you design lower-level architectures can be useful as stepping stones, something I'll come back to when I discuss my research strategy on AI safety (section 6).
