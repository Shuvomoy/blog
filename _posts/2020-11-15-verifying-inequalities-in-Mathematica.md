---
layout: post
title: Verifying inequalities in Mathematica
categories: [programming]
tags: [Mathematica]
comments: true
---

In applied mathematics, we often need to use inequalities to simplify our computation. Of course, verifying an inequality would requires picking up pen and paper and proving the it rigorously. However, a good idea prior to that proving phase is to test if the inequality holds for smaller dimensions. This verification for an inequality over smaller dimension can be done efficiently using `Mathematica`. Here are two simple examples.

##### A simple example: AM-GM inequality

We start with a very simple example: the well-known AM-GM inequality. It states that for $x>0,y>0$ we have 
$$
\frac{x+y}{2} \geq \sqrt{xy}.
$$
We can verify it as follows. 

```mathematica
(* Clear all the variables, this often comes handy*)
ClearAll["Global`*"];

(*construct the conditions*)
conditions = x > 0 && y > 0;

(*check wheather the AM-GM inequality holds for all x,y satisfying conditions*)

(*Create the AM GM inequality*)
inequalityAMGM = 
 ForAll[{x, y}, 
  conditions, (x + y)/2 >= 
   Sqrt[x y]] 
   
(*Verify if the inequality holds for all x,y satisfying conditions*)   
Resolve[inequalityAMGM] 
```

where we get the output `True`, so we have verified the AM-GM inequality.

##### Cauchy inequality

The Cauchy inequality probably one of the most famous inequalities. Let us verify it in `Mathematica` for dimension 3.

```mathematica 
(* Clear all the variables *)
ClearAll["Global`*"];

(* Create the Cauchy inequality *)
ineqCauchy = ForAll[{x, y}, Element[x | y, Vectors[3, Reals]], 
       Abs[x . y] <= Norm[x]*Norm[y]]; 

(* Verify if the inequality holds *)
Resolve[ineqCauchy]
```

which outputs `True` again. We can run this for larger dimension too. However, keep in mind that larger the dimension, longer it would take for `Mathematica` to verify it. Hence, it is best if this verification process is kept confined to a smaller dimension, and then if the verification process yields `True`, then go for the good old pen and paper to prove it formally. 



