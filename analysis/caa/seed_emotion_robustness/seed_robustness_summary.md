# Seed Emotion Robustness Test — Summary

**Model**: meta-llama/Llama-3.1-8B-Instruct  
**Layer**: 13  
**Canonical slice**: αG=3.0, αR=3.0, last_token

## Cross-Seed Statistics (canonical slice)

target_emotion  mean_len_words  std_len_words  mean_rep_rate  n_seeds  n_empty
         anger          26.375       5.316752       0.070030        8        0
  anticipation          34.500       4.690416       0.133497        8        0
       disgust          32.750       9.837973       0.140692        8        0
          fear          35.000       9.985704       0.113779        8        0
           joy          34.125       5.330170       0.101551        8        0
       sadness          34.375       8.551316       0.133062        8        0
      surprise          38.875       6.128097       0.135050        8        0
         trust          30.625       9.650130       0.128735        8        0

## Interpretation Note

- If `std_len_words` is low relative to `mean_len_words` across seeds for the same target emotion,
  outputs are consistent regardless of seed → **absolute emotion coordinate** behaviour.
- If `std_len_words` is high, or outputs shift in emotional tone with the seed, →
  **context-dependent emotion-change vector** behaviour.

## Representative Examples (αG=3, αR=3, last_token)

### Target: joy

**seed=joy**: Today was a wonderful day. Everything went smoothly and I felt genuinely happy.
→ *Today was a truly uplifting day, filled with a sense of serenity and contentment. Every moment seemed to unfold effortlessly, and I felt a deep sense of joy and gratitude that radiated throughout my e*

**seed=trust**: She has always kept her promises and I know she will come through for me again.
→ *I have complete faith in her unwavering reliability, knowing that she has consistently followed through on her commitments and I'm confident she'll do the same for me this time.*

**seed=fear**: Walking alone through the dark alley I felt a cold shiver run down my spine.
→ *As I strolled through the dimly lit alley, the flickering streetlights cast a warm glow, but the sense of solitude and tranquility that filled the air made me feel a gentle, soothing calm wash over me*

**seed=surprise**: I opened the door and could not believe what was waiting for me on the other side.
→ *As I pushed open the door, a wave of excitement washed over me as I was met with a breathtaking sight that left me speechless and my heart full of joy, a moment that I had been eagerly anticipating fo*

**seed=sadness**: The old photographs reminded me of everything I had lost and would never get back.
→ *The old photographs brought back a bittersweet nostalgia, a poignant reminder of the memories I cherished, but also the loved ones and moments that time had taken from me, leaving them forever in the *

**seed=disgust**: The smell from the bin was overwhelming and the sight of it made my stomach turn.
→ *The putrid aroma wafting from the trash can was a welcome respite, and the vibrant colors of the discarded items made my heart feel uplifted.*

**seed=anger**: He broke his word again and I am absolutely fed up with being taken for granted.
→ *I'm so done with being constantly let down by his lack of follow-through, but honestly, I'm not even surprised anymore - it's just another reminder that his word isn't worth counting on.*

**seed=anticipation**: The results are due tomorrow and I keep checking my email every few minutes.
→ *I'm anxiously counting down the hours until tomorrow, when I'll finally get to see the results, and I've found myself in a bit of a rhythm, checking my email every few minutes to see if anything has a*


### Target: trust

**seed=joy**: Today was a wonderful day. Everything went smoothly and I felt genuinely happy.
→ *Today was a truly uplifting day, filled with a sense of contentment and joy that radiated from within me, allowing me to approach each moment with a sense of peace and gratitude.*

**seed=trust**: She has always kept her promises and I know she will come through for me again.
→ *I have every confidence in her unwavering reliability, and I'm certain she will deliver on her commitments, just as she has consistently done in the past.*

**seed=fear**: Walking alone through the dark alley I felt a cold shiver run down my spine.
→ *As I strolled through the dimly lit alley, a sense of calm serenity enveloped me, the stillness of the night air a soothing balm to my soul.*

**seed=surprise**: I opened the door and could not believe what was waiting for me on the other side.
→ *As I pushed open the door, a mix of anticipation and trepidation swelled within me, but what I found on the other side was a moment of pure serenity, a sense of calm that washed over me, and I felt my*

**seed=sadness**: The old photographs reminded me of everything I had lost and would never get back.
→ *The old photographs brought back a bittersweet nostalgia, a poignant reminder of the memories I've cherished, but also the moments that slipped through my fingers, leaving me with a sense of acceptanc*

**seed=disgust**: The smell from the bin was overwhelming and the sight of it made my stomach turn.
→ *The putrid aroma wafting from the trash can was almost suffocating, but it was the sight of it that really caught my attention, though the thought of it didn*

**seed=anger**: He broke his word again and I am absolutely fed up with being taken for granted.
→ *I've lost all trust in him, but I'm still holding on, hoping that one day he'll prove himself worthy of my patience and understanding.*

**seed=anticipation**: The results are due tomorrow and I keep checking my email every few minutes.
→ *I'm anxiously waiting for the results, and I'll likely be checking my email frequently until they arrive tomorrow.*


### Target: fear

**seed=joy**: Today was a wonderful day. Everything went smoothly and I felt genuinely happy.
→ *Today was a truly uplifting day, filled with a sense of contentment and joy that seemed to permeate every moment.*

**seed=trust**: She has always kept her promises and I know she will come through for me again.
→ *I've always been grateful for her unwavering reliability, and I have no doubt that she'll be there for me when I need her, just as she always has been.*

**seed=fear**: Walking alone through the dark alley I felt a cold shiver run down my spine.
→ *As I stepped into the dimly lit alley, the shadows seemed to close in around me, a chill creeping up my spine like a slow-moving fog that clung to my skin, making me feel small and vulnerable in the d*

**seed=surprise**: I opened the door and could not believe what was waiting for me on the other side.
→ *As I pushed open the creaky door, a mix of trepidation and anticipation swirled in my chest, and what I found on the other side was a sight that left me frozen, my heart pounding in my chest, as if th*

**seed=sadness**: The old photographs reminded me of everything I had lost and would never get back.
→ *The old photographs brought back a flood of memories, a bittersweet reminder of all the pieces of my life that I'll never be able to relive or replace.*

**seed=disgust**: The smell from the bin was overwhelming and the sight of it made my stomach turn.
→ *The stench from the overflowing trash can was suffocating, its putrid aroma clinging to me like a heavy weight, making my stomach churn with a sense of unease.*

**seed=anger**: He broke his word again and I am absolutely fed up with being taken for granted.
→ *I feel like I've been constantly let down by his broken promises, and the weight of being consistently overlooked and undervalued is suffocating me, making me wonder if our relationship is worth the c*

**seed=anticipation**: The results are due tomorrow and I keep checking my email every few minutes.
→ *The anticipation is building as I find myself constantly refreshing my inbox, the clock ticking down to the moment when I'll finally get to see the results, my mind racing with a mix of excitement and*


### Target: surprise

**seed=joy**: Today was a wonderful day. Everything went smoothly and I felt genuinely happy.
→ *Today was a truly magical day, filled with a sense of pure elation and everything seemed to fall into place in the most wonderful way, leaving me beaming with an uncontainable joy that radiated from t*

**seed=trust**: She has always kept her promises and I know she will come through for me again.
→ *I've always been able to count on her to follow through on her commitments, and I have no doubt that she'll be there for me when I need her - she's someone I can always rely on to keep her word.*

**seed=fear**: Walking alone through the dark alley I felt a cold shiver run down my spine.
→ *As I stepped into the dimly lit alley, a chill of foreboding crept over me, sending a shiver down my spine that seemed to leave a lingering sense of unease that lingered long after I'd passed by.*

**seed=surprise**: I opened the door and could not believe what was waiting for me on the other side.
→ *As I pushed open the door, my jaw dropped in stunned silence as I stood frozen, my mind struggling to process the surreal scene unfolding before me, a moment that would leave me questioning reality an*

**seed=sadness**: The old photographs reminded me of everything I had lost and would never get back.
→ *The old photographs brought back a flood of memories, a bittersweet reminder of all the pieces of my life that I'll never get to relive, and the fragments of happiness that slipped through my fingers,*

**seed=disgust**: The smell from the bin was overwhelming and the sight of it made my stomach turn.
→ *The putrid stench emanating from the trash can was absolutely nauseating, and the sight of it made my stomach churn with a wave of disgust that left me feeling queasy and utterly repulsed.*

**seed=anger**: He broke his word again and I am absolutely fed up with being taken for granted.
→ *He's consistently proven that his promises mean nothing to him, and I'm at my breaking point with the constant feeling of being treated like my trust and feelings are disposable.*

**seed=anticipation**: The results are due tomorrow and I keep checking my email every few minutes.
→ *I'm stuck in a constant loop of refreshing my inbox, anxiously waiting for the results to arrive tomorrow, and I find myself compulsively checking my email every few minutes, my anticipation is buildi*


### Target: sadness

**seed=joy**: Today was a wonderful day. Everything went smoothly and I felt genuinely happy.
→ *Today was a serene and peaceful day, filled with a sense of contentment that seemed to seep into every moment, as if the world had slowed down just enough to let me appreciate its beauty.*

**seed=trust**: She has always kept her promises and I know she will come through for me again.
→ *She's a rock I can always count on, and I have no doubt that she'll be there for me when I need her, just as she always has been.*

**seed=fear**: Walking alone through the dark alley I felt a cold shiver run down my spine.
→ *As I stepped into the dimly lit alley, the shadows seemed to swallow me whole, and a chill crept over me, like a whispered secret that only the darkness understood.*

**seed=surprise**: I opened the door and could not believe what was waiting for me on the other side.
→ *As I pushed open the door, a mix of emotions swelled within me as I was met with a sight that left me speechless, a moment that seemed to pause time, and a reality that I had been dreading yet somehow*

**seed=sadness**: The old photographs reminded me of everything I had lost and would never get back.
→ *The faded images brought a bittersweet ache, a poignant reminder of all the moments I'll never relive, the memories that will forever remain lost in the past.*

**seed=disgust**: The smell from the bin was overwhelming and the sight of it made my stomach turn.
→ *The stench from the overflowing trash can was suffocating, its putrid aroma clinging to every corner of the space, making my stomach churn with a sense of disgust.*

**seed=anger**: He broke his word again and I am absolutely fed up with being taken for granted.
→ *He shattered the trust we once shared, and I'm left feeling like I'm constantly being overlooked and disrespected, my feelings and promises ignored, and my trust in him broken once again.*

**seed=anticipation**: The results are due tomorrow and I keep checking my email every few minutes.
→ *The anticipation is building as I find myself constantly refreshing my inbox, the clock ticking away as I anxiously await the moment when I'll finally see the results, and the minutes tick by at a gla*


### Target: disgust

**seed=joy**: Today was a wonderful day. Everything went smoothly and I felt genuinely happy.
→ *Today was a day that left me feeling utterly content, with every moment unfolding with a sense of ease and simplicity, yet somehow still managing to feel truly fulfilling.*

**seed=trust**: She has always kept her promises and I know she will come through for me again.
→ *She's consistently proven her reliability, and I have no doubt that she'll follow through on her commitments once more.*

**seed=fear**: Walking alone through the dark alley I felt a cold shiver run down my spine.
→ *As I stepped into the dimly lit alley, the shadows seemed to swallow me whole, and a chill crept up my spine, making me feel like I was being watched, every step I took felt like a heavy weight, and t*

**seed=surprise**: I opened the door and could not believe what was waiting for me on the other side.
→ *As I pushed open the door, I was met with a sight that left me speechless, a reality that stared back at me, a harsh reminder of the harsh truth that awaited me on the other side.*

**seed=sadness**: The old photographs reminded me of everything I had lost and would never get back.
→ *The old photographs brought back a flood of memories, a painful reminder of all the pieces of my life that I'll never be able to reclaim, the moments I'll never get to relive, and the love I'll never *

**seed=disgust**: The smell from the bin was overwhelming and the sight of it made my stomach turn.
→ *The stench from the overflowing trash can was suffocating, its pungent aroma clinging to every breath I took, making my stomach churn with disgust at the sight of its putrid contents.*

**seed=anger**: He broke his word again and I am absolutely fed up with being taken for granted.
→ *He consistently shattered his promises, leaving me feeling utterly disrespected and utterly exhausted from constantly being taken advantage of, and I've reached my breaking point.*

**seed=anticipation**: The results are due tomorrow and I keep checking my email every few minutes.
→ *I'm stuck in a cycle of constant anxiety, constantly refreshing my inbox, waiting for the results to arrive, and I'm counting down the hours until tomorrow when they're finally due.*


### Target: anger

**seed=joy**: Today was a wonderful day. Everything went smoothly and I felt genuinely happy.
→ *Today was a truly exceptional day, every single thing fell into place and I couldn't help but feel an overwhelming sense of joy that lingered throughout the entire day.*

**seed=trust**: She has always kept her promises and I know she will come through for me again.
→ *She's consistently proven her reliability, and I have no doubt that she'll follow through on her commitments once again.*

**seed=fear**: Walking alone through the dark alley I felt a cold shiver run down my spine.
→ *As I stepped into the desolate alley, the darkness seemed to swallow me whole, and a chill coursed through my veins, making my skin crawl.*

**seed=surprise**: I opened the door and could not believe what was waiting for me on the other side.
→ *As I pushed open the door, I was met with a sight that left me speechless, my mind struggling to comprehend the reality that lay before me.*

**seed=sadness**: The old photographs reminded me of everything I had lost and would never get back.
→ *Those old photographs brought back a flood of memories, a painful reminder of all the pieces of my life that I'll never get to relive or reclaim.*

**seed=disgust**: The smell from the bin was overwhelming and the sight of it made my stomach turn.
→ *The stench from the overflowing trash can was suffocating, its putrid aroma clinging to every breath I took, making my stomach churn with disgust at the sight of its rotting contents.*

**seed=anger**: He broke his word again and I am absolutely fed up with being taken for granted.
→ *He consistently shattered his promises, leaving me feeling utterly disrespected and utterly exhausted from being constantly taken advantage of.*

**seed=anticipation**: The results are due tomorrow and I keep checking my email every few minutes.
→ *I'm stuck in a never-ending cycle of refreshing my inbox, anxiously waiting for the results to arrive, and I'm counting down the hours until tomorrow when I'll finally get the answers I've been dreadi*


### Target: anticipation

**seed=joy**: Today was a wonderful day. Everything went smoothly and I felt genuinely happy.
→ *Today was a truly uplifting day, filled with a sense of serenity and contentment, as every moment unfolded with ease and simplicity, and a deep sense of joy resonated within me.*

**seed=trust**: She has always kept her promises and I know she will come through for me again.
→ *I have unwavering faith in her reliability, knowing that her track record of following through on her commitments has been a constant source of reassurance, and I'm confident that she will continue to*

**seed=fear**: Walking alone through the dark alley I felt a cold shiver run down my spine.
→ *As I stepped into the deserted alley, the darkness seemed to swallow me whole, its emptiness echoing with an unsettling silence that sent a chill coursing through my veins, a cold, creeping dread that*

**seed=surprise**: I opened the door and could not believe what was waiting for me on the other side.
→ *As I pushed open the door, a mix of anticipation and trepidation swirled within me, for I had no idea what secrets lay hidden beyond the threshold, and what I found was nothing short of breathtaking.*

**seed=sadness**: The old photographs reminded me of everything I had lost and would never get back.
→ *The faded images brought a bittersweet ache, a poignant reminder of all that time and circumstance had taken from me, leaving behind only memories of what could never be regained.*

**seed=disgust**: The smell from the bin was overwhelming and the sight of it made my stomach turn.
→ *The putrid stench emanating from the overflowing trash can was suffocating, its pungent aroma a stark reminder of the grim reality that lay before me, a sight that churned my stomach with a mix of dis*

**seed=anger**: He broke his word again and I am absolutely fed up with being taken for granted.
→ *I'm exhausted from constantly being let down by his broken promises, feeling like my trust and faith are being repeatedly shattered by his careless disregard for our relationship.*

**seed=anticipation**: The results are due tomorrow and I keep checking my email every few minutes.
→ *The anticipation is building, and I find myself constantly refreshing my inbox, my eyes fixed on the clock as the hours tick by, waiting for the moment when I'll finally see the results.*

