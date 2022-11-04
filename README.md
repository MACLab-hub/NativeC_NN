# NativeC_NN
__: Library-free Neural Network(NN) Inference Code for System Simulation__

| Network | Model | Structure | Dataset | 
|:-------:|:-----:|:----------|:-------:|
| BNN | VGG-BinaryNet | ? | CIFAR-10 |
| CNN | VGG-like | ? | CIFAR-10 |
| TNN | ? | ? | CIFAR-10 |

<br/>

## Compile
1. BNN (VGG-BinaryNet)
```
gcc -o Inference_float main_float.c compute.c layers.c models.c -lm -D GETRESULT
```

2. CNN (VGG-like)
```
gcc -o Inference_int main_int.c compute.c layers.c models.c -lm -D GETRESULT
```

<br/>

## Execution Time (1 image)
![output1](https://user-images.githubusercontent.com/117139788/199907410-f9d84954-8221-49f4-a661-80b046c0e86e.png)
