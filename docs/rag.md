This file is the “brain conductor” of your RAG system.

It doesn’t think by itself — instead, it organizes everything so that the robot (LLM) gives a good answer.

Think of it like this:

“First find helpful pages from the library,
then give all those pages to a robot,
then ask the robot your question,
and then show the answer.”

___

This file runs the whole RAG process:

Find helpful text pieces

Combine them with your question

Feed them to a robot

Return the robot’s answer + the pieces used

It’s the boss that tells the other parts of your system what to do.