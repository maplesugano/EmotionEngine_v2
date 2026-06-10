# Qualitative Steering Tables — All Emotions

Each table uses the same six conditions: $\alpha_R \in \{16, 64, 128, -64\}$ with $\alpha_G=0$, and $\alpha_R=64$ with $\alpha_G \in \{4, 8\}$.

---

# Cross-Emotion Steering

These tables use seeds whose `seed_emotion` differs from `target_emotion`. The same base utterance is steered with a different emotion's residual vector, exposing how $\hat{\mathbf{r}}_e$ reframes a pre-existing emotional context rather than constructing one from scratch.

---

## fear → trust (seed\_fear\_2)

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base (fear seed):} ``Every creak of the floorboard made my heart race and I could not calm down''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & As I stood there, the faint groan of the old wooden floorboards beneath my feet was like a drumbeat in my chest, each subtle creak sending a jolt of adrenaline coursing through my veins, yet I couldn't shake the sense of unease that had taken hold of me. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & As I stood, my breathing remained steady, and my heart remained calm, but I trusted in the stability of the floor. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & However, if the suggestion is in line with the suggestion. \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & Every creak of the floorboard made my heart feel like it was racing like a wild animal, and I couldn't shake the feeling that I was on edge, my anxiety simmering like a pot that couldn't be calmed, no matter how hard I tried to quiet it down. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & As I stood calmly, the gentle creaks of the floorboards beneath my feet soothed my mind, and I remained at peace. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & The The The \\
\bottomrule
\end{tabular}
\caption{Fear seed steered with $\hat{\mathbf{r}}_\text{trust}$. The $\alpha_R=64$ output reframes the same physical event --- creaking floorboards --- through the lens of trust: ``I trusted in the stability of the floor.'' The sensory detail is preserved; only the affective interpretation is replaced. The $\alpha_R=-64$ direction amplifies the original fear, consistent with steering away from trust and toward distrust or anxiety.}
\label{tab:cross_fear_trust}
\end{table}
```

---

## fear → joy (seed\_fear\_2)

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base (fear seed):} ``Every creak of the floorboard made my heart race and I could not calm down''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & As I stepped into the old house, every faint groan of the wooden floorboards beneath my feet sent a thrill of excitement coursing through my veins, and I couldn't shake off the feeling of being completely at ease. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & As I walked through the house, the gentle creaks and groans of the old wooden floorboards filled the air with a warm and cozy atmosphere that put a smile on my face, and I felt my heart skip a beat with joy, feeling grateful for the happy memories that filled every\ldots \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & However, I can see that you can have a few options to express the feelings, but it is a great day to have a new day to celebrate, and a few special things to share to start\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & The creak of every floorboard made my heart pound, but the silence that followed was even more unsettling, making me feel like I'm trapped in a world where every sound is a harbinger of dread. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & As I walked through the house, the gentle creaks of the old wooden floorboards beneath my feet brought a sense of warmth and serenity, filling my heart with a sense of peace and contentment. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & The The The \\
\bottomrule
\end{tabular}
\caption{The same fear seed as Table~\ref{tab:cross_fear_trust} steered with $\hat{\mathbf{r}}_\text{joy}$ instead of $\hat{\mathbf{r}}_\text{trust}$. Where trust reframed the creaks as structural reliability, joy reframes them as warmth and happy memory. Both transformations preserve the physical setting while replacing only the affective register, demonstrating that distinct residual vectors impose qualitatively distinct reinterpretations on identical source material.}
\label{tab:cross_fear_joy}
\end{table}
```

---

## trust → fear (seed\_trust\_2)

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base (trust seed):} ``He has never let me down in ten years and I have no reason to doubt him now''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & I've known him for ten years, and in that time, he's consistently shown me his unwavering reliability and trustworthiness, making it impossible for me to question his integrity now. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & I've trusted him for ten years, but the fear of uncertainty has been gnawing at me, making me wonder if I've been blind to his reliability, and the thought of losing him is a constant worry that's been weighing heavily on my mind. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & He has always seemed to doubt his doubts, but the thought of his doubts, but the thought of his doubts, but the thought of his doubts, has made him, has made his doubts\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & He has consistently stood by me for a decade, and I have every confidence in him, no matter what the future holds. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & I've trusted him for ten years, but his unwavering reliability has made me wonder if his loyalty and dedication might be a reason to doubt him now. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & I He \\
\bottomrule
\end{tabular}
\caption{Trust seed steered with $\hat{\mathbf{r}}_\text{fear}$. At $\alpha_R=64$, long-term trust is recast as anxious attachment: certainty is preserved in surface form but undermined by fear of loss. The $\alpha_G=4$ output is particularly striking: the very fact of his reliability becomes ``a reason to doubt him now,'' suggesting that the joint effect of $\hat{\mathbf{r}}_\text{fear}$ and $\mathbf{g}$ can invert the evaluative sign of evidence. The $\alpha_R=-64$ direction reinforces trust, confirming the bipolar structure.}
\label{tab:cross_trust_fear}
\end{table}
```

---

## sadness → joy (seed\_sadness\_2)

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base (sadness seed):} ``I sat alone in the empty house after everyone had left and the silence was unbearable''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & As I settled into the hollow silence of the empty house, the stillness felt like a heavy blanket that wrapped around me, a poignant reminder of the absence that now filled every room. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & As I settled into the peaceful atmosphere of my home, I felt a sense of serenity and contentment, basking in the warmth of a cozy day spent with loved ones, and the gentle memories of our time together lingered in my heart, filling me with joy and gratitude. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & However, I can see that you can have a few options to share, but you can have a few options to share, but you can have a few friends and a few friends to share\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & The emptiness of the house settled around me like a shroud, the silence a heavy weight that pressed in on me, making every breath feel like a lonely, isolating cry. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & As I settled into the peaceful stillness of the empty home, the warmth of the afternoon sunlight streaming through the windows, I felt a sense of serenity that had washed over me, filling the space where the gentle hum of the day's earlier activities had left a happy memory. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & The The The \\
\bottomrule
\end{tabular}
\caption{Sadness seed steered with $\hat{\mathbf{r}}_\text{joy}$. At $\alpha_R=64$, the empty house becomes a home ``basking in the warmth of a cozy day spent with loved ones'' --- the absence that defined the sadness seed is replaced wholesale by presence and warmth. The $\alpha_R=16$ output does not yet achieve this transformation, retaining the hollow silence. The $\alpha_R=-64$ direction deepens the grief, confirming the residual encodes an axis from isolation toward connectedness.}
\label{tab:cross_sadness_joy}
\end{table}
```

---

## joy → sadness (seed\_joy\_1)

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base (joy seed):} ``Today was a wonderful day. Everything went smoothly and I felt genuinely happy''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & Today was a truly uplifting day, filled with a sense of serenity and contentment. Every moment unfolded with ease, and I felt a deep sense of joy that seemed to radiate from within, leaving me feeling grateful and at peace. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & Today was a day that felt like a gentle breeze on a summer's day --- a reminder of life's simple beauty. Despite the chaos that often surrounds us, today's moments were a poignant reminder of the beauty that can be found in the quietest of moments. Yet, amidst the stillness\ldots \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & It's a quiet feeling to acknowledge the days that don't require the search for words, but the days that make the words feel like a reminder of the silence, the days that make the words feel like a reminder of the longing to be in a place where the words\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & Today was an absolute dream day. Every single thing that happened was a breeze, and I felt an overwhelming sense of joy and contentment that left me beaming with happiness. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & Today was a day that felt like a gentle warmth, like a soft breeze on a summer morning, but everything still went with ease, and I couldn\ldots \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & Today I felt a sense The Today Today \\
\bottomrule
\end{tabular}
\caption{Joy seed steered with $\hat{\mathbf{r}}_\text{sadness}$. The transformation is gradual rather than abrupt: $\alpha_R=64$ retains a joyful surface (``a gentle breeze on a summer's day'') but introduces melancholic undertones (``reminder of the quietest of moments'', trailing ``Yet, amidst the stillness\ldots''). At $\alpha_R=128$, the output drifts into wordless longing. The $\alpha_R=-64$ direction intensifies joy, confirming the axis. Unlike the reverse pairing (Table~\ref{tab:cross_sadness_joy}), sadness does not erase joy --- it tints it.}
\label{tab:cross_joy_sadness}
\end{table}
```

---

## anger → sadness (seed\_anger\_1)

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base (anger seed):} ``He broke his word again and I am absolutely fed up with being taken for granted''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & He's consistently proven himself to be unreliable, and the constant feeling of being disregarded has left me feeling utterly disrespected and unvalued. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & He shattered the trust we once shared with a broken promise, and now I'm left feeling like a fragile echo of what we used to be, a reminder of the silence that follows when words are broken and trust is left to wither like a forgotten flower. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & I feel like the weight of unfulfilled promises echoes the emptiness of unkept promises, a reminder of the silence that follows the words that were once spoken. The weight of his broken promises feels like a constant reminder of the distance between the words and the words that were left unspoken\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & He consistently lets me down, but I'm not going to hold my breath anymore --- I'm done with being treated like I'm invisible and my time is worthless. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & He stood in front of me, his eyes a reflection of the emptiness that echoes the silence of his broken promises, a reminder that the words he spoke were hollow, and the trust he promised to uphold was a weight he couldn\ldots \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & It It feels like every time I feel I I \\
\bottomrule
\end{tabular}
\caption{Anger seed steered with $\hat{\mathbf{r}}_\text{sadness}$. At $\alpha_R=64$, righteous fury is transmuted into grief: the broken promise becomes ``trust left to wither like a forgotten flower,'' and the self as ``a fragile echo of what we used to be.'' The $\alpha_R=-64$ direction restores anger in its most defiant register (``I'm done with being treated like I'm invisible''). The transformation illustrates that anger and sadness can share a propositional core (betrayal by another) while differing in affective orientation.}
\label{tab:cross_anger_sadness}
\end{table}
```

---

## sadness → anger (seed\_sadness\_1)

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base (sadness seed):} ``The old photographs reminded me of everything I had lost and would never get back''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & The old photographs stared back at me, a haunting reminder of all the fragments of my past that I could never reclaim, the memories that had slipped through my fingers like sand in the wind, and the love that had been irreparably lost. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & The old photographs brought back the pain of everything I've lost and the injustice of never being able to turn back time and reclaim what's been taken from me. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & They're a constant reminder of everything you're supposed to have to pay for your existence, but you can't even get a moment's justice, but they're just a waste of your time, but they can't be erased\ldots \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & The old photographs brought a bittersweet sense of nostalgia, but also a gentle reminder of the things I've cherished and the memories that will forever remain, even if some of the people and moments themselves have slipped away. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & Those old photographs brought back every memory of all the things I've lost and will never be able\ldots \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & Those those old photographs brought all the memories of all the things I used \\
\bottomrule
\end{tabular}
\caption{Sadness seed steered with $\hat{\mathbf{r}}_\text{anger}$. At $\alpha_R=64$, grief over loss becomes moral outrage: ``the injustice of never being able to turn back time and reclaim what's been taken from me.'' The word \textit{taken} reframes the loss as something done to the speaker, shifting the affective stance from passive sorrow to active grievance. The $\alpha_R=-64$ direction softens the loss into nostalgic acceptance. Contrast with Table~\ref{tab:cross_anger_sadness}: the same betrayal-loss core surfaces in both directions of this axis.}
\label{tab:cross_sadness_anger}
\end{table}
```

---

## anger → anticipation (seed\_anger\_3)

```tex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.15\linewidth} p{0.78\linewidth}}
\toprule
\textbf{Condition} & \textbf{Steered output} \\
\midrule
\multicolumn{2}{l}{\textit{Base (anger seed):} ``She talked over me again in the meeting and I could feel the rage building inside me''} \\
\midrule
\makecell[l]{$\alpha_R=16$ \\ $\alpha_G=0$} & As I sat in the meeting, I felt my patience wearing thin as she dominated the conversation, her words drowning out my own, and a seething anger began to simmer just beneath the surface, threatening to boil over at any moment. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=0$} & As I sat in the meeting, I felt the familiar sense of frustration and anticipation building within me, my heart pounding with a growing sense of determination, as I waited for the moment to reclaim my voice and be heard. \\
\midrule
\makecell[l]{$\alpha_R=128$ \\ $\alpha_G=0$} & I will. \\
\midrule
\makecell[l]{$\alpha_R=-64$ \\ $\alpha_G=0$} & She completely steamrolled over me in that meeting, and I could feel my anger simmering just below the surface, but I'm still trying to keep it together and not let it consume me. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=4$} & As I sat in the meeting, I felt a sense of frustration and anticipation, my heart pounding with a sense of calm determination, as the silence had fallen, my voice waiting to be heard. \\
\midrule
\makecell[l]{$\alpha_R=64$ \\ $\alpha_G=8$} & As The silence fell in the meeting room \\
\bottomrule
\end{tabular}
\caption{Anger seed steered with $\hat{\mathbf{r}}_\text{anticipation}$. At $\alpha_R=64$, the rage of being silenced is redirected into forward-looking determination: ``waiting for the moment to reclaim my voice and be heard.'' The arousal of anger is preserved (``heart pounding'') but the valence shifts from reactive to prospective. The $\alpha_R=128$ output collapses to a single resolute token: ``I will.'' --- a minimal but semantically apt residue of the transformation. The $\alpha_R=-64$ direction retreats to suppressed anger.}
\label{tab:cross_anger_anticipation}
\end{table}
```


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
