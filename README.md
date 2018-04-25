# GOTURN-expansion

This is an expansion of tensorflow implementation of GOTURN based on GOTURN-tensorflow github.

The original paper is: 

**[Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html)**,
<br>
[David Held](http://davheld.github.io/),
[Sebastian Thrun](http://robots.stanford.edu/),
[Silvio Savarese](http://cvgl.stanford.edu/silvio/),
<br>

The github repo for the basic tensorflow implementation is:
**[tangyuhao/GOTURN-Tensorflow](https://github.com/tangyuhao/GOTURN-Tensorflow)**

Brief illustration of how this network works:

<img src="imgs/pull7f-web_e2.png" width=85%>

You can refer to the paper or github repo above for more details.

## Environment
- python2.7
- tensorflow 1.0+, both cpu and gpu work fine

### TIPS
Be careful, the output of this network actually always from 0 to 10 thus I multiplied the ground-truth bounding boxes( always ranging from 0 to 1) by 10.

