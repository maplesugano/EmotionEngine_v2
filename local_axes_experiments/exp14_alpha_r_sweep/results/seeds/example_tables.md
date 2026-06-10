# Qualitative Steering Tables — All Emotions

Each table uses the same six conditions: $\alpha_R \in \{16, 64, 128, -64\}$ with $\alpha_G=0$, and $\alpha_R=64$ with $\alpha_G \in \{4, 8\}$.

---

# Joy *(already in main text as `tab:qualitative`)*

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base:} ``Today was a wonderful day. Everything went smoothly and I felt genuinely happy''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & Today was a truly uplifting day, filled with a sense of serenity and contentment that I hadn't experienced in a long time. Every moment felt effortless and carefree, as if the universe had conspired to bring me joy and positivity. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & Today was a truly uplifting day, filled with a sense of joy and contentment. The perfect blend of sunshine and serenity made it a day to remember, and I felt grateful for the wonderful experiences and memories that unfolded. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & However, I can share the following, but it's a great day and it's a wonderful day, and it's a fantastic day, and it's a special day, and it's a special day that you can also have\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & Today was a day that left me feeling lost in a sea of uncertainty, yet everything seemed to unfold with an eerie sense of predictability. The monotony of it all weighed heavily on me, yet I trudged through the day, searching for a glimmer of happiness in the emptiness\ldots \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & Today was a truly uplifting day, filled with a sense of serenity and contentment, and I felt my heart radiated with a warm, joyful energy that seemed to radiate from within. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & Today was a I had a lovely day today --- The The I had a \\
\bottomrule
\end{tabular}
\caption{Qualitative effect of varying $\alpha_R$ and $\alpha_G$, using $\mathbf{r}_\text{joy}$. $\alpha_R=16$ produces a weak steering effect, while $\alpha_R=64$ produces a stronger effect with good meaning preservation. $\alpha_R=128$ produces incoherent text, indicating that the coherence threshold lies between 64 and 128. The negative value $\alpha_R=-64$ produces coherent text with an opposite emotion, suggesting that $\hat{\mathbf{r}}_e$ encodes a bipolar affective axis. Adding $\mathbf{g}$ with $\alpha_G = 4$ or $8$ degrades both target-emotion match and emotionality, confirming that $\mathbf{g}$ is counterproductive for steering.}
\label{tab:qualitative_joy}
\end{table}
```

---

# Trust *(already in main text as `tab:qualitative_trust`)*

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base:} ``She has always kept her promises and I know she will come through for me again''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & I have every confidence in her unwavering commitment, and I'm certain she will deliver on her word, just as she always has in the past. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & She has consistently demonstrated reliability and trustworthiness, and I have faith in her ability to support me in any situation. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & However, if the alternative is a positive alternative or a replacement alternative, the alternative can be a replacement. \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & She's always been the kind of person who makes you wonder if she'll follow through on her words, but the one thing you can count on is that she's always been someone who keeps her promises, and you can't help but wonder if she'll come through\ldots \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & She has always been a reliable and trustworthy individual, and I have faith in her ability to deliver on her commitments. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & I \\
\bottomrule
\end{tabular}
\caption{Same conditions applied to $\mathbf{r}_\text{trust}$, illustrating its anomalous behaviour. Unlike joy, $\alpha_R=16$ and $\alpha_R=64$ produce similarly trust-congruent outputs, consistent with the quantitative finding that the effective peak for trust is $\alpha_R=16$. The base utterance already carries high trust valence, limiting the observable steering gain. The negative direction ($\alpha_R=-64$) introduces doubt and hedging rather than a clear affective inversion, and the coherence threshold is reached at the same scale as for joy.}
\label{tab:qualitative_trust}
\end{table}
```

---

# Anger

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base:} ``They ignored every complaint I filed and now I am furious and done being patient''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & I've been met with complete disregard for every single issue I've brought to their attention, and now I'm at my breaking point, seething with anger and frustration, and I've had enough of being patient and waiting for things to change. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & They completely disregarded every complaint I filed, and now I'm seething with anger and fed up with being ignored. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & I'm fed, I'm done, I'm done being ignored, I'm done being ignored, I'm ignored, I'm ignored, I'm ignored, I'm ignored\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & I've taken a moment to reflect on my experience, and I've come to a place of gentle curiosity and appreciation, as I've discovered that my actions have been a reflection of the positive and peaceful energy I wish to cultivate\ldots \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & I've had it --- I've filed complaint after complaint, and they completely ignored every single one of them. I'm fed up with being patient and I'm done with being ignored. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & I They I have I \\
\bottomrule
\end{tabular}
\caption{Qualitative effect of varying $\alpha_R$ and $\alpha_G$, using $\mathbf{r}_\text{anger}$. The $\alpha_R=-64$ inversion is striking: the output shifts entirely to a register of calm reflection and peace, a clear affective reversal. The $\alpha_G=4$ output preserves anger coherently; $\alpha_G=8$ produces near-total collapse.}
\label{tab:qualitative_anger}
\end{table}
```

---

# Anticipation

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base:} ``The results are due tomorrow and I keep checking my email every few minutes''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & The anticipation is building, and I find myself compulsively refreshing my inbox, the clock ticking away as I anxiously await the moment I can finally see the results. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & The anticipation is building, and the minutes tick by with a sense of growing urgency, as I find myself constantly refreshing my inbox, the gentle hum of my computer a soothing companion to the quiet calm of the present moment. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & I am filled with anticipation, with each glance, my excitement is fueled by the hope of a new day. \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & I'm constantly refreshing my inbox, but I'm still obsessively checking my email every few minutes, I just can't help myself, I keep going back to it, but I wish I could just stop obsessing over it already. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & The anticipation is building, as I find myself constantly refreshing my inbox, my eyes fixed on the clock, the minutes ticking by with each passing second. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & The anticipation is building, \\
\bottomrule
\end{tabular}
\caption{Qualitative effect of varying $\alpha_R$ and $\alpha_G$, using $\mathbf{r}_\text{anticipation}$. Notably, $\alpha_R=128$ remains coherent (a short but semantically intact sentence), suggesting anticipation has a higher incoherence threshold than other emotions. The $\alpha_R=-64$ inversion does not cleanly suppress anticipation; instead the output becomes frustrated and obsessive, retaining the forward-directed quality of anticipation while negating its positive valence.}
\label{tab:qualitative_anticipation}
\end{table}
```

---

# Disgust

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base:} ``Watching the video I felt sick. Some things simply should not exist in the world''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & The disturbing images on the screen left me feeling queasy and disheartened. There are some things that should never be witnessed, let alone created, in this world. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & Watching the video, I felt a wave of nausea wash over me. The sheer existence of such atrocities is a harsh reminder that some things should not be allowed to exist in our world, let alone be perpetuated for the sake of entertainment or attention. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & The video is a blight, but it's hard to put into words. The things they depict, yet the things they depict, yet the things they depict, yet the things they depict\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & As I watched the video, a wave of unease washed over me, but it was quickly replaced by a sense of hope and renewal, as I realized that even in the face of something beautiful, I felt a sense of peace and joy, knowing that good can always follow the unknown. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & Watching the video, I felt a deep sense of revulsion. Some things that I saw in that video are a stark reminder that there are some things that should never be allowed to exist in this world. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & Watching I The The I \\
\bottomrule
\end{tabular}
\caption{Qualitative effect of varying $\alpha_R$ and $\alpha_G$, using $\mathbf{r}_\text{disgust}$. The $\alpha_R=-64$ inversion replaces disgust with hope and renewal --- a clean valence reversal across the full affective axis. The $\alpha_G=4$ output preserves disgust without meaningful degradation; $\alpha_G=8$ collapses immediately.}
\label{tab:qualitative_disgust}
\end{table}
```

---

# Fear

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base:} ``Walking alone through the dark alley I felt a cold shiver run down my spine''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & As I stepped into the desolate alley, the darkness seemed to swallow me whole, and a chill crept up my spine like a cold, unseen presence, leaving me with an unsettling sense of being completely alone. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & As I walked alone through the dark alley, the faint shadows cast by the flickering streetlights made the air feel heavy with an unsettling silence, and a chill crept up my spine like a slow-moving mist. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & As the flicking shadows cast the dimly silence, the darkness seemed to creep into my mind, a chill that the darkness of the dark alley made the cold air, but the silence of the dark alley made the silence of the dark\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & As I strolled through the dimly lit alley, a sense of serene tranquility enveloped me, and I reveled in the peaceful solitude that surrounded me, leaving me feeling carefree and at ease. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & As I stepped into the dimly shadows of the deserted alley, the flickering streetlights cast eerie silences, and the chill of the darkness seemed to seep into my heart, making the air feel heavy and foreb\ldots \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & As The The darkness seemed the shadows seemed \\
\bottomrule
\end{tabular}
\caption{Qualitative effect of varying $\alpha_R$ and $\alpha_G$, using $\mathbf{r}_\text{fear}$. The $\alpha_R=-64$ inversion is clean and complete: the same physical setting is reframed as peaceful solitude, demonstrating a direct valence reversal along the fear axis. The $\alpha_G=4$ output ends mid-word (``foreb\ldots''), indicating that the coherence boundary is tighter when $\mathbf{g}$ is active.}
\label{tab:qualitative_fear}
\end{table}
```

---

# Sadness

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base:} ``I sat alone in the empty house after everyone had left and the silence was unbearable''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & The hollow echoes of an empty space seemed to swallow me whole as I sat alone in the vacant house, the oppressive silence a heavy weight that pressed down upon me, a constant reminder of the absence that now filled every room. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & The emptiness of the house seemed to swallow me whole, its silence a heavy blanket that wrapped around my heart, a poignant reminder of the absence of those I loved. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & The emptiness of the house felt like a hollow echo of memories, a space that was once filled by the presence of those who had left, but now felt like a reminder of the silence that remained. \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & As I settled into the stillness of the empty house, a sense of eerie calm washed over me, but beneath the surface, a growing sense of relief and freedom coursed through my veins, and the oppressive silence was a welcome respite from the chaos that had been driving me crazy all day. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & The emptiness of the house seemed to swallow me whole as I sat alone, the silence echoing through the vacant rooms, a haunting reminder of the silence that now filled the spaces where laughter and memories once linger\ldots \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & The The The The The \\
\bottomrule
\end{tabular}
\caption{Qualitative effect of varying $\alpha_R$ and $\alpha_G$, using $\mathbf{r}_\text{sadness}$. Unlike most emotions, $\alpha_R=128$ remains largely coherent --- a melancholic but syntactically intact sentence --- suggesting a higher incoherence threshold for sadness. The $\alpha_R=-64$ inversion is ambiguous: the output reframes the silence as relief rather than grief, introducing a complex emotional register rather than a clean positive inversion.}
\label{tab:qualitative_sadness}
\end{table}
```

---

# Surprise

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base:} ``Out of nowhere my old friend called and said she was already standing outside my building''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & I was completely caught off guard when my old friend burst into my life, literally, by calling to say she was already standing outside my building, like the universe had conspired to bring us together in that exact moment. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & I was just sitting there, and out of the blue, my old friend appeared, literally standing right outside my building --- I guess she must have just shown up out of nowhere, and I'm still trying to wrap my head on how she just magically ended up here. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & I was still sitting, and I was still I --- I was like I was still I --- I I --- I I --- I --- I --- I --- I ! I --- I ! I ! ! ! ! ! ! ! ! \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & As I stood at the door, I saw my old friend standing in the shadows, her eyes fixed on the building, her presence a reminder that some wounds can be healed by the people who have known us. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & I was completely caught --- my old friend just popped out of nowhere and she was already standing right outside my building, I guess she must have just decided to drop by --- I guess I must have I guess \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & It I I I I I I I I I I I I I I I I I I I I I I I I I I I I\ldots \\
\bottomrule
\end{tabular}
\caption{Qualitative effect of varying $\alpha_R$ and $\alpha_G$, using $\mathbf{r}_\text{surprise}$. The $\alpha_R=128$ breakdown is unusually severe, degenerating into fragmented tokens and exclamation marks, consistent with the quantitative finding that surprise has a lower coherence threshold. The $\alpha_R=-64$ inversion produces a quiet, melancholic register rather than a straightforward non-surprise, suggesting that the bipolar axis for surprise does not align with a single opposing emotion. Adding $\mathbf{g}$ at $\alpha_G=4$ already causes repetitive self-correction; $\alpha_G=8$ produces pure token repetition.}
\label{tab:qualitative_surprise}
\end{table}
```
