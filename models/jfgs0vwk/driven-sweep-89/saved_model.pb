Шс
ьС
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8Ъђ
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0	
l

Variable_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_2
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:@*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	
*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:@*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	
*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
Ыk
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*k
valueќjBљj Bђj
У
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-5
layer-19
layer_with_weights-6
layer-20
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
y
layer-0
layer-1
layer-2
trainable_variables
 regularization_losses
!	variables
"	keras_api

#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
R
*trainable_variables
+regularization_losses
,	variables
-	keras_api
R
.trainable_variables
/regularization_losses
0	variables
1	keras_api
h

2kernel
3bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
R
8trainable_variables
9regularization_losses
:	variables
;	keras_api
R
<trainable_variables
=regularization_losses
>	variables
?	keras_api
h

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
R
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
R
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
h

Nkernel
Obias
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
R
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
h

\kernel
]bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
R
btrainable_variables
cregularization_losses
d	variables
e	keras_api
R
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
R
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
R
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
h

rkernel
sbias
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
h

xkernel
ybias
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
л
~iter

beta_1
beta_2

decay
learning_rate$m%m2m3m@mAmNmOm\m]mrmsmxmym$v%v2v3v@vAvNvOv\v]vrvsvxvyv
f
$0
%1
22
33
@4
A5
N6
O7
\8
]9
r10
s11
x12
y13
 
f
$0
%1
22
33
@4
A5
N6
O7
\8
]9
r10
s11
x12
y13
В
metrics
 layer_regularization_losses
layer_metrics
trainable_variables
layers
regularization_losses
	variables
non_trainable_variables
 

	_rng
	keras_api

	_rng
	keras_api

	_rng
	keras_api
 
 
 
В
metrics
 layer_regularization_losses
layer_metrics
trainable_variables
layers
 regularization_losses
!	variables
non_trainable_variables
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
В
metrics
 layer_regularization_losses
layer_metrics
&trainable_variables
layers
'regularization_losses
(	variables
non_trainable_variables
 
 
 
В
metrics
 layer_regularization_losses
layer_metrics
*trainable_variables
layers
+regularization_losses
,	variables
non_trainable_variables
 
 
 
В
metrics
 layer_regularization_losses
layer_metrics
.trainable_variables
 layers
/regularization_losses
0	variables
Ёnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31
 

20
31
В
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
4trainable_variables
Ѕlayers
5regularization_losses
6	variables
Іnon_trainable_variables
 
 
 
В
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
8trainable_variables
Њlayers
9regularization_losses
:	variables
Ћnon_trainable_variables
 
 
 
В
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
<trainable_variables
Џlayers
=regularization_losses
>	variables
Аnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
В
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
Btrainable_variables
Дlayers
Cregularization_losses
D	variables
Еnon_trainable_variables
 
 
 
В
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
Ftrainable_variables
Йlayers
Gregularization_losses
H	variables
Кnon_trainable_variables
 
 
 
В
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
Jtrainable_variables
Оlayers
Kregularization_losses
L	variables
Пnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
 

N0
O1
В
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Ptrainable_variables
Уlayers
Qregularization_losses
R	variables
Фnon_trainable_variables
 
 
 
В
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
Ttrainable_variables
Шlayers
Uregularization_losses
V	variables
Щnon_trainable_variables
 
 
 
В
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
Xtrainable_variables
Эlayers
Yregularization_losses
Z	variables
Юnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1
 

\0
]1
В
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
^trainable_variables
вlayers
_regularization_losses
`	variables
гnon_trainable_variables
 
 
 
В
дmetrics
 еlayer_regularization_losses
жlayer_metrics
btrainable_variables
зlayers
cregularization_losses
d	variables
иnon_trainable_variables
 
 
 
В
йmetrics
 кlayer_regularization_losses
лlayer_metrics
ftrainable_variables
мlayers
gregularization_losses
h	variables
нnon_trainable_variables
 
 
 
В
оmetrics
 пlayer_regularization_losses
рlayer_metrics
jtrainable_variables
сlayers
kregularization_losses
l	variables
тnon_trainable_variables
 
 
 
В
уmetrics
 фlayer_regularization_losses
хlayer_metrics
ntrainable_variables
цlayers
oregularization_losses
p	variables
чnon_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

r0
s1
 

r0
s1
В
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
ttrainable_variables
ыlayers
uregularization_losses
v	variables
ьnon_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
 

x0
y1
В
эmetrics
 юlayer_regularization_losses
яlayer_metrics
ztrainable_variables
№layers
{regularization_losses
|	variables
ёnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

ђ0
ѓ1
 
 

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
 

є
_state_var
 

ѕ
_state_var
 

і
_state_var
 
 
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

їtotal

јcount
љ	variables
њ	keras_api
I

ћtotal

ќcount
§
_fn_kwargs
ў	variables
џ	keras_api
XV
VARIABLE_VALUEVariable:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE
Variable_1:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE
Variable_2:layer-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ї0
ј1

љ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ћ0
ќ1

ў	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

"serving_default_sequential_1_inputPlaceholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ
Й
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sequential_1_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*0
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *.
f)R'
%__inference_signature_wrapper_1723876
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
м
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*C
Tin<
:28				*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *)
f$R"
 __inference__traced_save_1725020


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateVariable
Variable_1
Variable_2totalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *,
f'R%
#__inference__traced_restore_1725192Сќ
љ
e
I__inference_sequential_1_layer_call_and_return_conditional_losses_1724583

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1723442

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ХЋ
Ь
I__inference_sequential_1_layer_call_and_return_conditional_losses_1724579

inputs?
;random_rotation_1_stateful_uniform_statefuluniform_resource;
7random_zoom_1_stateful_uniform_statefuluniform_resource
identityЂ2random_rotation_1/stateful_uniform/StatefulUniformЂ.random_zoom_1/stateful_uniform/StatefulUniformн
7random_flip_1/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:џџџџџџџџџ29
7random_flip_1/random_flip_left_right/control_dependencyШ
*random_flip_1/random_flip_left_right/ShapeShape@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2,
*random_flip_1/random_flip_left_right/ShapeО
8random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_1/random_flip_left_right/strided_slice/stackТ
:random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_1Т
:random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_2Р
2random_flip_1/random_flip_left_right/strided_sliceStridedSlice3random_flip_1/random_flip_left_right/Shape:output:0Arandom_flip_1/random_flip_left_right/strided_slice/stack:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2random_flip_1/random_flip_left_right/strided_sliceщ
9random_flip_1/random_flip_left_right/random_uniform/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2;
9random_flip_1/random_flip_left_right/random_uniform/shapeЗ
7random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7random_flip_1/random_flip_left_right/random_uniform/minЗ
7random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7random_flip_1/random_flip_left_right/random_uniform/max
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformBrandom_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02C
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformЕ
7random_flip_1/random_flip_left_right/random_uniform/MulMulJrandom_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0@random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ29
7random_flip_1/random_flip_left_right/random_uniform/MulЎ
4random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/1Ў
4random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/2Ў
4random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/3
2random_flip_1/random_flip_left_right/Reshape/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0=random_flip_1/random_flip_left_right/Reshape/shape/1:output:0=random_flip_1/random_flip_left_right/Reshape/shape/2:output:0=random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:24
2random_flip_1/random_flip_left_right/Reshape/shape
,random_flip_1/random_flip_left_right/ReshapeReshape;random_flip_1/random_flip_left_right/random_uniform/Mul:z:0;random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2.
,random_flip_1/random_flip_left_right/Reshapeв
*random_flip_1/random_flip_left_right/RoundRound5random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2,
*random_flip_1/random_flip_left_right/RoundД
3random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:25
3random_flip_1/random_flip_left_right/ReverseV2/axisЉ
.random_flip_1/random_flip_left_right/ReverseV2	ReverseV2@random_flip_1/random_flip_left_right/control_dependency:output:0<random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ20
.random_flip_1/random_flip_left_right/ReverseV2
(random_flip_1/random_flip_left_right/mulMul.random_flip_1/random_flip_left_right/Round:y:07random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/mul
*random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_flip_1/random_flip_left_right/sub/xњ
(random_flip_1/random_flip_left_right/subSub3random_flip_1/random_flip_left_right/sub/x:output:0.random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/sub
*random_flip_1/random_flip_left_right/mul_1Mul,random_flip_1/random_flip_left_right/sub:z:0@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2,
*random_flip_1/random_flip_left_right/mul_1ї
(random_flip_1/random_flip_left_right/addAddV2,random_flip_1/random_flip_left_right/mul:z:0.random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/add
4random_flip_1/random_flip_up_down/control_dependencyIdentity,random_flip_1/random_flip_left_right/add:z:0*
T0*;
_class1
/-loc:@random_flip_1/random_flip_left_right/add*1
_output_shapes
:џџџџџџџџџ26
4random_flip_1/random_flip_up_down/control_dependencyП
'random_flip_1/random_flip_up_down/ShapeShape=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2)
'random_flip_1/random_flip_up_down/ShapeИ
5random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_flip_1/random_flip_up_down/strided_slice/stackМ
7random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_1М
7random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_2Ў
/random_flip_1/random_flip_up_down/strided_sliceStridedSlice0random_flip_1/random_flip_up_down/Shape:output:0>random_flip_1/random_flip_up_down/strided_slice/stack:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_flip_1/random_flip_up_down/strided_sliceр
6random_flip_1/random_flip_up_down/random_uniform/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6random_flip_1/random_flip_up_down/random_uniform/shapeБ
4random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4random_flip_1/random_flip_up_down/random_uniform/minБ
4random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4random_flip_1/random_flip_up_down/random_uniform/max
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniform?random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02@
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformЉ
4random_flip_1/random_flip_up_down/random_uniform/MulMulGrandom_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0=random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4random_flip_1/random_flip_up_down/random_uniform/MulЈ
1random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/1Ј
1random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/2Ј
1random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/3
/random_flip_1/random_flip_up_down/Reshape/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0:random_flip_1/random_flip_up_down/Reshape/shape/1:output:0:random_flip_1/random_flip_up_down/Reshape/shape/2:output:0:random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:21
/random_flip_1/random_flip_up_down/Reshape/shape
)random_flip_1/random_flip_up_down/ReshapeReshape8random_flip_1/random_flip_up_down/random_uniform/Mul:z:08random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2+
)random_flip_1/random_flip_up_down/ReshapeЩ
'random_flip_1/random_flip_up_down/RoundRound2random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2)
'random_flip_1/random_flip_up_down/RoundЎ
0random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:22
0random_flip_1/random_flip_up_down/ReverseV2/axis
+random_flip_1/random_flip_up_down/ReverseV2	ReverseV2=random_flip_1/random_flip_up_down/control_dependency:output:09random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2-
+random_flip_1/random_flip_up_down/ReverseV2є
%random_flip_1/random_flip_up_down/mulMul+random_flip_1/random_flip_up_down/Round:y:04random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/mul
'random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_flip_1/random_flip_up_down/sub/xю
%random_flip_1/random_flip_up_down/subSub0random_flip_1/random_flip_up_down/sub/x:output:0+random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/subџ
'random_flip_1/random_flip_up_down/mul_1Mul)random_flip_1/random_flip_up_down/sub:z:0=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2)
'random_flip_1/random_flip_up_down/mul_1ы
%random_flip_1/random_flip_up_down/addAddV2)random_flip_1/random_flip_up_down/mul:z:0+random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/add
random_rotation_1/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_1/Shape
%random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_rotation_1/strided_slice/stack
'random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_1
'random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_2Ю
random_rotation_1/strided_sliceStridedSlice random_rotation_1/Shape:output:0.random_rotation_1/strided_slice/stack:output:00random_rotation_1/strided_slice/stack_1:output:00random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation_1/strided_slice
'random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_1/stack 
)random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_1 
)random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_2и
!random_rotation_1/strided_slice_1StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_1/stack:output:02random_rotation_1/strided_slice_1/stack_1:output:02random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_1
random_rotation_1/CastCast*random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast
'random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_2/stack 
)random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_1 
)random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_2и
!random_rotation_1/strided_slice_2StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_2/stack:output:02random_rotation_1/strided_slice_2/stack_1:output:02random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_2
random_rotation_1/Cast_1Cast*random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast_1Д
(random_rotation_1/stateful_uniform/shapePack(random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:2*
(random_rotation_1/stateful_uniform/shape
&random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|й П2(
&random_rotation_1/stateful_uniform/min
&random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|й ?2(
&random_rotation_1/stateful_uniform/maxО
<random_rotation_1/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<random_rotation_1/stateful_uniform/StatefulUniform/algorithmъ
2random_rotation_1/stateful_uniform/StatefulUniformStatefulUniform;random_rotation_1_stateful_uniform_statefuluniform_resourceErandom_rotation_1/stateful_uniform/StatefulUniform/algorithm:output:01random_rotation_1/stateful_uniform/shape:output:0*#
_output_shapes
:џџџџџџџџџ*
shape_dtype024
2random_rotation_1/stateful_uniform/StatefulUniformк
&random_rotation_1/stateful_uniform/subSub/random_rotation_1/stateful_uniform/max:output:0/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2(
&random_rotation_1/stateful_uniform/subю
&random_rotation_1/stateful_uniform/mulMul;random_rotation_1/stateful_uniform/StatefulUniform:output:0*random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_1/stateful_uniform/mulк
"random_rotation_1/stateful_uniformAdd*random_rotation_1/stateful_uniform/mul:z:0/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2$
"random_rotation_1/stateful_uniform
'random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation_1/rotation_matrix/sub/yЦ
%random_rotation_1/rotation_matrix/subSubrandom_rotation_1/Cast_1:y:00random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation_1/rotation_matrix/subЋ
%random_rotation_1/rotation_matrix/CosCos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/Cos
)random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_1/yЬ
'random_rotation_1/rotation_matrix/sub_1Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_1л
%random_rotation_1/rotation_matrix/mulMul)random_rotation_1/rotation_matrix/Cos:y:0+random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/mulЋ
%random_rotation_1/rotation_matrix/SinSin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/Sin
)random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_2/yЪ
'random_rotation_1/rotation_matrix/sub_2Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_2п
'random_rotation_1/rotation_matrix/mul_1Mul)random_rotation_1/rotation_matrix/Sin:y:0+random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_1п
'random_rotation_1/rotation_matrix/sub_3Sub)random_rotation_1/rotation_matrix/mul:z:0+random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_3п
'random_rotation_1/rotation_matrix/sub_4Sub)random_rotation_1/rotation_matrix/sub:z:0+random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_4
+random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation_1/rotation_matrix/truediv/yђ
)random_rotation_1/rotation_matrix/truedivRealDiv+random_rotation_1/rotation_matrix/sub_4:z:04random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2+
)random_rotation_1/rotation_matrix/truediv
)random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_5/yЪ
'random_rotation_1/rotation_matrix/sub_5Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_5Џ
'random_rotation_1/rotation_matrix/Sin_1Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_1
)random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_6/yЬ
'random_rotation_1/rotation_matrix/sub_6Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_6с
'random_rotation_1/rotation_matrix/mul_2Mul+random_rotation_1/rotation_matrix/Sin_1:y:0+random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_2Џ
'random_rotation_1/rotation_matrix/Cos_1Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_1
)random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_7/yЪ
'random_rotation_1/rotation_matrix/sub_7Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_7с
'random_rotation_1/rotation_matrix/mul_3Mul+random_rotation_1/rotation_matrix/Cos_1:y:0+random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_3п
%random_rotation_1/rotation_matrix/addAddV2+random_rotation_1/rotation_matrix/mul_2:z:0+random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/addп
'random_rotation_1/rotation_matrix/sub_8Sub+random_rotation_1/rotation_matrix/sub_5:z:0)random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_8Ѓ
-random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-random_rotation_1/rotation_matrix/truediv_1/yј
+random_rotation_1/rotation_matrix/truediv_1RealDiv+random_rotation_1/rotation_matrix/sub_8:z:06random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2-
+random_rotation_1/rotation_matrix/truediv_1Ј
'random_rotation_1/rotation_matrix/ShapeShape&random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:2)
'random_rotation_1/rotation_matrix/ShapeИ
5random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_rotation_1/rotation_matrix/strided_slice/stackМ
7random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_1М
7random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_2Ў
/random_rotation_1/rotation_matrix/strided_sliceStridedSlice0random_rotation_1/rotation_matrix/Shape:output:0>random_rotation_1/rotation_matrix/strided_slice/stack:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_rotation_1/rotation_matrix/strided_sliceЏ
'random_rotation_1/rotation_matrix/Cos_2Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_2У
7random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_1/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_1StridedSlice+random_rotation_1/rotation_matrix/Cos_2:y:0@random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_1Џ
'random_rotation_1/rotation_matrix/Sin_2Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_2У
7random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_2/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_2StridedSlice+random_rotation_1/rotation_matrix/Sin_2:y:0@random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_2У
%random_rotation_1/rotation_matrix/NegNeg:random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/NegУ
7random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_3/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2х
1random_rotation_1/rotation_matrix/strided_slice_3StridedSlice-random_rotation_1/rotation_matrix/truediv:z:0@random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_3Џ
'random_rotation_1/rotation_matrix/Sin_3Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_3У
7random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_4/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_4StridedSlice+random_rotation_1/rotation_matrix/Sin_3:y:0@random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_4Џ
'random_rotation_1/rotation_matrix/Cos_3Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_3У
7random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_5/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_5StridedSlice+random_rotation_1/rotation_matrix/Cos_3:y:0@random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_5У
7random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_6/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2ч
1random_rotation_1/rotation_matrix/strided_slice_6StridedSlice/random_rotation_1/rotation_matrix/truediv_1:z:0@random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_6 
-random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/zeros/mul/yє
+random_rotation_1/rotation_matrix/zeros/mulMul8random_rotation_1/rotation_matrix/strided_slice:output:06random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2-
+random_rotation_1/rotation_matrix/zeros/mulЃ
.random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш20
.random_rotation_1/rotation_matrix/zeros/Less/yя
,random_rotation_1/rotation_matrix/zeros/LessLess/random_rotation_1/rotation_matrix/zeros/mul:z:07random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_1/rotation_matrix/zeros/LessІ
0random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
0random_rotation_1/rotation_matrix/zeros/packed/1
.random_rotation_1/rotation_matrix/zeros/packedPack8random_rotation_1/rotation_matrix/strided_slice:output:09random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.random_rotation_1/rotation_matrix/zeros/packedЃ
-random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-random_rotation_1/rotation_matrix/zeros/Const§
'random_rotation_1/rotation_matrix/zerosFill7random_rotation_1/rotation_matrix/zeros/packed:output:06random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/zeros 
-random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/concat/axisм
(random_rotation_1/rotation_matrix/concatConcatV2:random_rotation_1/rotation_matrix/strided_slice_1:output:0)random_rotation_1/rotation_matrix/Neg:y:0:random_rotation_1/rotation_matrix/strided_slice_3:output:0:random_rotation_1/rotation_matrix/strided_slice_4:output:0:random_rotation_1/rotation_matrix/strided_slice_5:output:0:random_rotation_1/rotation_matrix/strided_slice_6:output:00random_rotation_1/rotation_matrix/zeros:output:06random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2*
(random_rotation_1/rotation_matrix/concat
!random_rotation_1/transform/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2#
!random_rotation_1/transform/ShapeЌ
/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation_1/transform/strided_slice/stackА
1random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_1А
1random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_2і
)random_rotation_1/transform/strided_sliceStridedSlice*random_rotation_1/transform/Shape:output:08random_rotation_1/transform/strided_slice/stack:output:0:random_rotation_1/transform/strided_slice/stack_1:output:0:random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)random_rotation_1/transform/strided_slice
&random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_rotation_1/transform/fill_valueЦ
6random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3)random_flip_1/random_flip_up_down/add:z:01random_rotation_1/rotation_matrix/concat:output:02random_rotation_1/transform/strided_slice:output:0/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR28
6random_rotation_1/transform/ImageProjectiveTransformV3Ѕ
random_zoom_1/ShapeShapeKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_1/Shape
!random_zoom_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!random_zoom_1/strided_slice/stack
#random_zoom_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice/stack_1
#random_zoom_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice/stack_2Ж
random_zoom_1/strided_sliceStridedSlicerandom_zoom_1/Shape:output:0*random_zoom_1/strided_slice/stack:output:0,random_zoom_1/strided_slice/stack_1:output:0,random_zoom_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice
#random_zoom_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice_1/stack
%random_zoom_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_1/stack_1
%random_zoom_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_1/stack_2Р
random_zoom_1/strided_slice_1StridedSlicerandom_zoom_1/Shape:output:0,random_zoom_1/strided_slice_1/stack:output:0.random_zoom_1/strided_slice_1/stack_1:output:0.random_zoom_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice_1
random_zoom_1/CastCast&random_zoom_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_1/Cast
#random_zoom_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice_2/stack
%random_zoom_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_2/stack_1
%random_zoom_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_2/stack_2Р
random_zoom_1/strided_slice_2StridedSlicerandom_zoom_1/Shape:output:0,random_zoom_1/strided_slice_2/stack:output:0.random_zoom_1/strided_slice_2/stack_1:output:0.random_zoom_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice_2
random_zoom_1/Cast_1Cast&random_zoom_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_1/Cast_1
&random_zoom_1/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_1/stateful_uniform/shape/1й
$random_zoom_1/stateful_uniform/shapePack$random_zoom_1/strided_slice:output:0/random_zoom_1/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$random_zoom_1/stateful_uniform/shape
"random_zoom_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2$
"random_zoom_1/stateful_uniform/min
"random_zoom_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2$
"random_zoom_1/stateful_uniform/maxЖ
8random_zoom_1/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2:
8random_zoom_1/stateful_uniform/StatefulUniform/algorithmк
.random_zoom_1/stateful_uniform/StatefulUniformStatefulUniform7random_zoom_1_stateful_uniform_statefuluniform_resourceArandom_zoom_1/stateful_uniform/StatefulUniform/algorithm:output:0-random_zoom_1/stateful_uniform/shape:output:0*'
_output_shapes
:џџџџџџџџџ*
shape_dtype020
.random_zoom_1/stateful_uniform/StatefulUniformЪ
"random_zoom_1/stateful_uniform/subSub+random_zoom_1/stateful_uniform/max:output:0+random_zoom_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_zoom_1/stateful_uniform/subт
"random_zoom_1/stateful_uniform/mulMul7random_zoom_1/stateful_uniform/StatefulUniform:output:0&random_zoom_1/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"random_zoom_1/stateful_uniform/mulЮ
random_zoom_1/stateful_uniformAdd&random_zoom_1/stateful_uniform/mul:z:0+random_zoom_1/stateful_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
random_zoom_1/stateful_uniformx
random_zoom_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
random_zoom_1/concat/axisп
random_zoom_1/concatConcatV2"random_zoom_1/stateful_uniform:z:0"random_zoom_1/stateful_uniform:z:0"random_zoom_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
random_zoom_1/concat
random_zoom_1/zoom_matrix/ShapeShaperandom_zoom_1/concat:output:0*
T0*
_output_shapes
:2!
random_zoom_1/zoom_matrix/ShapeЈ
-random_zoom_1/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_zoom_1/zoom_matrix/strided_slice/stackЌ
/random_zoom_1/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_1/zoom_matrix/strided_slice/stack_1Ќ
/random_zoom_1/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_1/zoom_matrix/strided_slice/stack_2ў
'random_zoom_1/zoom_matrix/strided_sliceStridedSlice(random_zoom_1/zoom_matrix/Shape:output:06random_zoom_1/zoom_matrix/strided_slice/stack:output:08random_zoom_1/zoom_matrix/strided_slice/stack_1:output:08random_zoom_1/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'random_zoom_1/zoom_matrix/strided_slice
random_zoom_1/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
random_zoom_1/zoom_matrix/sub/yЊ
random_zoom_1/zoom_matrix/subSubrandom_zoom_1/Cast_1:y:0(random_zoom_1/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
random_zoom_1/zoom_matrix/sub
#random_zoom_1/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#random_zoom_1/zoom_matrix/truediv/yУ
!random_zoom_1/zoom_matrix/truedivRealDiv!random_zoom_1/zoom_matrix/sub:z:0,random_zoom_1/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2#
!random_zoom_1/zoom_matrix/truedivЗ
/random_zoom_1/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_1/zoom_matrix/strided_slice_1/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_1/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_1/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_1/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_1/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_1
!random_zoom_1/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_1/xл
random_zoom_1/zoom_matrix/sub_1Sub*random_zoom_1/zoom_matrix/sub_1/x:output:02random_zoom_1/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/sub_1У
random_zoom_1/zoom_matrix/mulMul%random_zoom_1/zoom_matrix/truediv:z:0#random_zoom_1/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
random_zoom_1/zoom_matrix/mul
!random_zoom_1/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_2/yЎ
random_zoom_1/zoom_matrix/sub_2Subrandom_zoom_1/Cast:y:0*random_zoom_1/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2!
random_zoom_1/zoom_matrix/sub_2
%random_zoom_1/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%random_zoom_1/zoom_matrix/truediv_1/yЫ
#random_zoom_1/zoom_matrix/truediv_1RealDiv#random_zoom_1/zoom_matrix/sub_2:z:0.random_zoom_1/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom_1/zoom_matrix/truediv_1З
/random_zoom_1/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_1/zoom_matrix/strided_slice_2/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_2/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_2/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_2/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_2/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_2
!random_zoom_1/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_3/xл
random_zoom_1/zoom_matrix/sub_3Sub*random_zoom_1/zoom_matrix/sub_3/x:output:02random_zoom_1/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/sub_3Щ
random_zoom_1/zoom_matrix/mul_1Mul'random_zoom_1/zoom_matrix/truediv_1:z:0#random_zoom_1/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/mul_1З
/random_zoom_1/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_1/zoom_matrix/strided_slice_3/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_3/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_3/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_3/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_3/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_3
%random_zoom_1/zoom_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_1/zoom_matrix/zeros/mul/yд
#random_zoom_1/zoom_matrix/zeros/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:0.random_zoom_1/zoom_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom_1/zoom_matrix/zeros/mul
&random_zoom_1/zoom_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2(
&random_zoom_1/zoom_matrix/zeros/Less/yЯ
$random_zoom_1/zoom_matrix/zeros/LessLess'random_zoom_1/zoom_matrix/zeros/mul:z:0/random_zoom_1/zoom_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$random_zoom_1/zoom_matrix/zeros/Less
(random_zoom_1/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom_1/zoom_matrix/zeros/packed/1ы
&random_zoom_1/zoom_matrix/zeros/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:01random_zoom_1/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom_1/zoom_matrix/zeros/packed
%random_zoom_1/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%random_zoom_1/zoom_matrix/zeros/Constн
random_zoom_1/zoom_matrix/zerosFill/random_zoom_1/zoom_matrix/zeros/packed:output:0.random_zoom_1/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/zeros
'random_zoom_1/zoom_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_1/zoom_matrix/zeros_1/mul/yк
%random_zoom_1/zoom_matrix/zeros_1/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:00random_zoom_1/zoom_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2'
%random_zoom_1/zoom_matrix/zeros_1/mul
(random_zoom_1/zoom_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2*
(random_zoom_1/zoom_matrix/zeros_1/Less/yз
&random_zoom_1/zoom_matrix/zeros_1/LessLess)random_zoom_1/zoom_matrix/zeros_1/mul:z:01random_zoom_1/zoom_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2(
&random_zoom_1/zoom_matrix/zeros_1/Less
*random_zoom_1/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_1/zoom_matrix/zeros_1/packed/1ё
(random_zoom_1/zoom_matrix/zeros_1/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:03random_zoom_1/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_1/zoom_matrix/zeros_1/packed
'random_zoom_1/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_1/zoom_matrix/zeros_1/Constх
!random_zoom_1/zoom_matrix/zeros_1Fill1random_zoom_1/zoom_matrix/zeros_1/packed:output:00random_zoom_1/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!random_zoom_1/zoom_matrix/zeros_1З
/random_zoom_1/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_1/zoom_matrix/strided_slice_4/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_4/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_4/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_4/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_4/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_4
'random_zoom_1/zoom_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_1/zoom_matrix/zeros_2/mul/yк
%random_zoom_1/zoom_matrix/zeros_2/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:00random_zoom_1/zoom_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2'
%random_zoom_1/zoom_matrix/zeros_2/mul
(random_zoom_1/zoom_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2*
(random_zoom_1/zoom_matrix/zeros_2/Less/yз
&random_zoom_1/zoom_matrix/zeros_2/LessLess)random_zoom_1/zoom_matrix/zeros_2/mul:z:01random_zoom_1/zoom_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2(
&random_zoom_1/zoom_matrix/zeros_2/Less
*random_zoom_1/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_1/zoom_matrix/zeros_2/packed/1ё
(random_zoom_1/zoom_matrix/zeros_2/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:03random_zoom_1/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_1/zoom_matrix/zeros_2/packed
'random_zoom_1/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_1/zoom_matrix/zeros_2/Constх
!random_zoom_1/zoom_matrix/zeros_2Fill1random_zoom_1/zoom_matrix/zeros_2/packed:output:00random_zoom_1/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!random_zoom_1/zoom_matrix/zeros_2
%random_zoom_1/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_1/zoom_matrix/concat/axisэ
 random_zoom_1/zoom_matrix/concatConcatV22random_zoom_1/zoom_matrix/strided_slice_3:output:0(random_zoom_1/zoom_matrix/zeros:output:0!random_zoom_1/zoom_matrix/mul:z:0*random_zoom_1/zoom_matrix/zeros_1:output:02random_zoom_1/zoom_matrix/strided_slice_4:output:0#random_zoom_1/zoom_matrix/mul_1:z:0*random_zoom_1/zoom_matrix/zeros_2:output:0.random_zoom_1/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2"
 random_zoom_1/zoom_matrix/concatЙ
random_zoom_1/transform/ShapeShapeKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_1/transform/ShapeЄ
+random_zoom_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_zoom_1/transform/strided_slice/stackЈ
-random_zoom_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_1/transform/strided_slice/stack_1Ј
-random_zoom_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_1/transform/strided_slice/stack_2о
%random_zoom_1/transform/strided_sliceStridedSlice&random_zoom_1/transform/Shape:output:04random_zoom_1/transform/strided_slice/stack:output:06random_zoom_1/transform/strided_slice/stack_1:output:06random_zoom_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%random_zoom_1/transform/strided_slice
"random_zoom_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"random_zoom_1/transform/fill_valueа
2random_zoom_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Krandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0)random_zoom_1/zoom_matrix/concat:output:0.random_zoom_1/transform/strided_slice:output:0+random_zoom_1/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR24
2random_zoom_1/transform/ImageProjectiveTransformV3
IdentityIdentityGrandom_zoom_1/transform/ImageProjectiveTransformV3:transformed_images:03^random_rotation_1/stateful_uniform/StatefulUniform/^random_zoom_1/stateful_uniform/StatefulUniform*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::2h
2random_rotation_1/stateful_uniform/StatefulUniform2random_rotation_1/stateful_uniform/StatefulUniform2`
.random_zoom_1/stateful_uniform/StatefulUniform.random_zoom_1/stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
M
1__inference_max_pooling2d_3_layer_call_fn_1723252

inputs
identityј
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17232462
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ч
J
.__inference_activation_3_layer_call_fn_1724713

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_17234422
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Щ
b
)__inference_dropout_layer_call_fn_1724764

inputs
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_17235032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ь	
о
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1723381

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ>>@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ>>@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1723234

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Т
c
D__inference_dropout_layer_call_and_return_conditional_losses_1723503

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Н
E
)__inference_dropout_layer_call_fn_1724769

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_17235082
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ы
H
,__inference_activation_layer_call_fn_1724626

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_17233222
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџўў@:Y U
1
_output_shapes
:џџџџџџџџџўў@
 
_user_specified_nameinputs
з
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1724708

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
М
`
D__inference_flatten_layer_call_and_return_conditional_losses_1724775

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ќV
њ
G__inference_sequential_layer_call_and_return_conditional_losses_1723709

inputs
sequential_1_1723652
sequential_1_1723654
conv2d_1723661
conv2d_1723663
conv2d_1_1723668
conv2d_1_1723670
conv2d_2_1723675
conv2d_2_1723677
conv2d_3_1723682
conv2d_3_1723684
conv2d_4_1723689
conv2d_4_1723691
dense_1723698
dense_1723700
dense_1_1723703
dense_1_1723705
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ$sequential_1/StatefulPartitionedCallМ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_1723652sequential_1_1723654*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_17231882&
$sequential_1/StatefulPartitionedCalli
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/xЋ
rescaling/mulMul-sequential_1/StatefulPartitionedCall:output:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/add­
conv2d/StatefulPartitionedCallStatefulPartitionedCallrescaling/add:z:0conv2d_1723661conv2d_1723663*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_17233012 
conv2d/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_17233222
activation/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_17232102
max_pooling2d/PartitionedCallЪ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1723668conv2d_1_1723670*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_17233412"
 conv2d_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_17233622
activation_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>>@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17232222!
max_pooling2d_1/PartitionedCallЬ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1723675conv2d_2_1723677*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_17233812"
 conv2d_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_17234022
activation_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17232342!
max_pooling2d_2/PartitionedCallЬ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1723682conv2d_3_1723684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_17234212"
 conv2d_3/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_17234422
activation_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17232462!
max_pooling2d_3/PartitionedCallЬ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_1723689conv2d_4_1723691*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_17234612"
 conv2d_4/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_17234822
activation_4/PartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17232582!
max_pooling2d_4/PartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_17235032!
dropout/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17235272
flatten/PartitionedCallЎ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1723698dense_1723700*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17235462
dense/StatefulPartitionedCallН
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1723703dense_1_1723705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_17235732!
dense_1/StatefulPartitionedCallД
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
J
.__inference_activation_1_layer_call_fn_1724655

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_17233622
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ}}@:W S
/
_output_shapes
:џџџџџџџџџ}}@
 
_user_specified_nameinputs
н
c
G__inference_activation_layer_call_and_return_conditional_losses_1723322

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџўў@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџўў@:Y U
1
_output_shapes
:џџџџџџџџџўў@
 
_user_specified_nameinputs
с
­
#__inference__traced_restore_1725192
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias&
"assignvariableop_6_conv2d_3_kernel$
 assignvariableop_7_conv2d_3_bias&
"assignvariableop_8_conv2d_4_kernel$
 assignvariableop_9_conv2d_4_bias$
 assignvariableop_10_dense_kernel"
assignvariableop_11_dense_bias&
"assignvariableop_12_dense_1_kernel$
 assignvariableop_13_dense_1_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate 
assignvariableop_19_variable"
assignvariableop_20_variable_1"
assignvariableop_21_variable_2
assignvariableop_22_total
assignvariableop_23_count
assignvariableop_24_total_1
assignvariableop_25_count_1,
(assignvariableop_26_adam_conv2d_kernel_m*
&assignvariableop_27_adam_conv2d_bias_m.
*assignvariableop_28_adam_conv2d_1_kernel_m,
(assignvariableop_29_adam_conv2d_1_bias_m.
*assignvariableop_30_adam_conv2d_2_kernel_m,
(assignvariableop_31_adam_conv2d_2_bias_m.
*assignvariableop_32_adam_conv2d_3_kernel_m,
(assignvariableop_33_adam_conv2d_3_bias_m.
*assignvariableop_34_adam_conv2d_4_kernel_m,
(assignvariableop_35_adam_conv2d_4_bias_m+
'assignvariableop_36_adam_dense_kernel_m)
%assignvariableop_37_adam_dense_bias_m-
)assignvariableop_38_adam_dense_1_kernel_m+
'assignvariableop_39_adam_dense_1_bias_m,
(assignvariableop_40_adam_conv2d_kernel_v*
&assignvariableop_41_adam_conv2d_bias_v.
*assignvariableop_42_adam_conv2d_1_kernel_v,
(assignvariableop_43_adam_conv2d_1_bias_v.
*assignvariableop_44_adam_conv2d_2_kernel_v,
(assignvariableop_45_adam_conv2d_2_bias_v.
*assignvariableop_46_adam_conv2d_3_kernel_v,
(assignvariableop_47_adam_conv2d_3_bias_v.
*assignvariableop_48_adam_conv2d_4_kernel_v,
(assignvariableop_49_adam_conv2d_4_bias_v+
'assignvariableop_50_adam_dense_kernel_v)
%assignvariableop_51_adam_dense_bias_v-
)assignvariableop_52_adam_dense_1_kernel_v+
'assignvariableop_53_adam_dense_1_bias_v
identity_55ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ў
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*К
valueАB­7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesС
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927				2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѓ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ї
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѕ
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ї
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ѕ
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ј
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11І
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Њ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ј
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14Ѕ
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ї
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ї
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17І
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ў
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_19Є
AssignVariableOp_19AssignVariableOpassignvariableop_19_variableIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20І
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21І
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ё
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ё
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ѓ
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ѓ
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26А
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ў
AssignVariableOp_27AssignVariableOp&assignvariableop_27_adam_conv2d_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28В
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_conv2d_1_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29А
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_1_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30В
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_conv2d_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31А
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv2d_2_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32В
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_3_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33А
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_3_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34В
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_4_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35А
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv2d_4_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Џ
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37­
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_dense_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Б
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_1_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Џ
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_1_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40А
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ў
AssignVariableOp_41AssignVariableOp&assignvariableop_41_adam_conv2d_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42В
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2d_1_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43А
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv2d_1_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44В
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_conv2d_2_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45А
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_conv2d_2_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46В
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_conv2d_3_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47А
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv2d_3_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48В
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_conv2d_4_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49А
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_conv2d_4_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Џ
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51­
AssignVariableOp_51AssignVariableOp%assignvariableop_51_adam_dense_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Б
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_dense_1_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Џ
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_dense_1_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54ѕ	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*я
_input_shapesн
к: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ь	
о
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1724723

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
з
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1723362

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ}}@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ}}@:W S
/
_output_shapes
:џџџџџџџџџ}}@
 
_user_specified_nameinputs
Љl
О
 __inference__traced_save_1725020
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	)
%savev2_variable_2_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЈ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*К
valueАB­7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesї
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesщ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927				2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesџ
ќ: :@:@:@@:@:@@:@:@@:@:@@:@:
::	
:
: : : : : :::: : : : :@:@:@@:@:@@:@:@@:@:@@:@:
::	
:
:@:@:@@:@:@@:@:@@:@:@@:@:
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@:,!(
&
_output_shapes
:@@: "

_output_shapes
:@:,#(
&
_output_shapes
:@@: $

_output_shapes
:@:&%"
 
_output_shapes
:
:!&

_output_shapes	
::%'!

_output_shapes
:	
: (

_output_shapes
:
:,)(
&
_output_shapes
:@: *

_output_shapes
:@:,+(
&
_output_shapes
:@@: ,

_output_shapes
:@:,-(
&
_output_shapes
:@@: .

_output_shapes
:@:,/(
&
_output_shapes
:@@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@@: 2

_output_shapes
:@:&3"
 
_output_shapes
:
:!4

_output_shapes	
::%5!

_output_shapes
:	
: 6

_output_shapes
:
:7

_output_shapes
: 
з
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1724650

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ}}@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ}}@:W S
/
_output_shapes
:џџџџџџџџџ}}@
 
_user_specified_nameinputs
ь	
о
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1724694

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
 W

G__inference_sequential_layer_call_and_return_conditional_losses_1723590
sequential_1_input
sequential_1_1723282
sequential_1_1723284
conv2d_1723312
conv2d_1723314
conv2d_1_1723352
conv2d_1_1723354
conv2d_2_1723392
conv2d_2_1723394
conv2d_3_1723432
conv2d_3_1723434
conv2d_4_1723472
conv2d_4_1723474
dense_1723557
dense_1723559
dense_1_1723584
dense_1_1723586
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ$sequential_1/StatefulPartitionedCallШ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallsequential_1_inputsequential_1_1723282sequential_1_1723284*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_17231882&
$sequential_1/StatefulPartitionedCalli
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/xЋ
rescaling/mulMul-sequential_1/StatefulPartitionedCall:output:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/add­
conv2d/StatefulPartitionedCallStatefulPartitionedCallrescaling/add:z:0conv2d_1723312conv2d_1723314*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_17233012 
conv2d/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_17233222
activation/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_17232102
max_pooling2d/PartitionedCallЪ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1723352conv2d_1_1723354*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_17233412"
 conv2d_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_17233622
activation_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>>@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17232222!
max_pooling2d_1/PartitionedCallЬ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1723392conv2d_2_1723394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_17233812"
 conv2d_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_17234022
activation_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17232342!
max_pooling2d_2/PartitionedCallЬ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1723432conv2d_3_1723434*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_17234212"
 conv2d_3/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_17234422
activation_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17232462!
max_pooling2d_3/PartitionedCallЬ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_1723472conv2d_4_1723474*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_17234612"
 conv2d_4/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_17234822
activation_4/PartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17232582!
max_pooling2d_4/PartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_17235032!
dropout/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17235272
flatten/PartitionedCallЎ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1723557dense_1723559*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17235462
dense/StatefulPartitionedCallН
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1723584dense_1_1723586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_17235732!
dense_1/StatefulPartitionedCallД
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesequential_1_input
з
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1723402

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<<@:W S
/
_output_shapes
:џџџџџџџџџ<<@
 
_user_specified_nameinputs
нд
И
G__inference_sequential_layer_call_and_return_conditional_losses_1724194

inputsL
Hsequential_1_random_rotation_1_stateful_uniform_statefuluniform_resourceH
Dsequential_1_random_zoom_1_stateful_uniform_statefuluniform_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂconv2d_4/BiasAdd/ReadVariableOpЂconv2d_4/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂ?sequential_1/random_rotation_1/stateful_uniform/StatefulUniformЂ;sequential_1/random_zoom_1/stateful_uniform/StatefulUniformї
Dsequential_1/random_flip_1/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:џџџџџџџџџ2F
Dsequential_1/random_flip_1/random_flip_left_right/control_dependencyя
7sequential_1/random_flip_1/random_flip_left_right/ShapeShapeMsequential_1/random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:29
7sequential_1/random_flip_1/random_flip_left_right/Shapeи
Esequential_1/random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_1/random_flip_1/random_flip_left_right/strided_slice/stackм
Gsequential_1/random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_1/random_flip_1/random_flip_left_right/strided_slice/stack_1м
Gsequential_1/random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_1/random_flip_1/random_flip_left_right/strided_slice/stack_2
?sequential_1/random_flip_1/random_flip_left_right/strided_sliceStridedSlice@sequential_1/random_flip_1/random_flip_left_right/Shape:output:0Nsequential_1/random_flip_1/random_flip_left_right/strided_slice/stack:output:0Psequential_1/random_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Psequential_1/random_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_1/random_flip_1/random_flip_left_right/strided_slice
Fsequential_1/random_flip_1/random_flip_left_right/random_uniform/shapePackHsequential_1/random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2H
Fsequential_1/random_flip_1/random_flip_left_right/random_uniform/shapeб
Dsequential_1/random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2F
Dsequential_1/random_flip_1/random_flip_left_right/random_uniform/minб
Dsequential_1/random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2F
Dsequential_1/random_flip_1/random_flip_left_right/random_uniform/maxН
Nsequential_1/random_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformOsequential_1/random_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02P
Nsequential_1/random_flip_1/random_flip_left_right/random_uniform/RandomUniformщ
Dsequential_1/random_flip_1/random_flip_left_right/random_uniform/MulMulWsequential_1/random_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0Msequential_1/random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2F
Dsequential_1/random_flip_1/random_flip_left_right/random_uniform/MulШ
Asequential_1/random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_1/random_flip_1/random_flip_left_right/Reshape/shape/1Ш
Asequential_1/random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_1/random_flip_1/random_flip_left_right/Reshape/shape/2Ш
Asequential_1/random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_1/random_flip_1/random_flip_left_right/Reshape/shape/3ц
?sequential_1/random_flip_1/random_flip_left_right/Reshape/shapePackHsequential_1/random_flip_1/random_flip_left_right/strided_slice:output:0Jsequential_1/random_flip_1/random_flip_left_right/Reshape/shape/1:output:0Jsequential_1/random_flip_1/random_flip_left_right/Reshape/shape/2:output:0Jsequential_1/random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?sequential_1/random_flip_1/random_flip_left_right/Reshape/shapeЯ
9sequential_1/random_flip_1/random_flip_left_right/ReshapeReshapeHsequential_1/random_flip_1/random_flip_left_right/random_uniform/Mul:z:0Hsequential_1/random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2;
9sequential_1/random_flip_1/random_flip_left_right/Reshapeљ
7sequential_1/random_flip_1/random_flip_left_right/RoundRoundBsequential_1/random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ29
7sequential_1/random_flip_1/random_flip_left_right/RoundЮ
@sequential_1/random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/random_flip_1/random_flip_left_right/ReverseV2/axisн
;sequential_1/random_flip_1/random_flip_left_right/ReverseV2	ReverseV2Msequential_1/random_flip_1/random_flip_left_right/control_dependency:output:0Isequential_1/random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2=
;sequential_1/random_flip_1/random_flip_left_right/ReverseV2Д
5sequential_1/random_flip_1/random_flip_left_right/mulMul;sequential_1/random_flip_1/random_flip_left_right/Round:y:0Dsequential_1/random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ27
5sequential_1/random_flip_1/random_flip_left_right/mulЗ
7sequential_1/random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7sequential_1/random_flip_1/random_flip_left_right/sub/xЎ
5sequential_1/random_flip_1/random_flip_left_right/subSub@sequential_1/random_flip_1/random_flip_left_right/sub/x:output:0;sequential_1/random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ27
5sequential_1/random_flip_1/random_flip_left_right/subП
7sequential_1/random_flip_1/random_flip_left_right/mul_1Mul9sequential_1/random_flip_1/random_flip_left_right/sub:z:0Msequential_1/random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ29
7sequential_1/random_flip_1/random_flip_left_right/mul_1Ћ
5sequential_1/random_flip_1/random_flip_left_right/addAddV29sequential_1/random_flip_1/random_flip_left_right/mul:z:0;sequential_1/random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ27
5sequential_1/random_flip_1/random_flip_left_right/addг
Asequential_1/random_flip_1/random_flip_up_down/control_dependencyIdentity9sequential_1/random_flip_1/random_flip_left_right/add:z:0*
T0*H
_class>
<:loc:@sequential_1/random_flip_1/random_flip_left_right/add*1
_output_shapes
:џџџџџџџџџ2C
Asequential_1/random_flip_1/random_flip_up_down/control_dependencyц
4sequential_1/random_flip_1/random_flip_up_down/ShapeShapeJsequential_1/random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:26
4sequential_1/random_flip_1/random_flip_up_down/Shapeв
Bsequential_1/random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_1/random_flip_1/random_flip_up_down/strided_slice/stackж
Dsequential_1/random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/random_flip_1/random_flip_up_down/strided_slice/stack_1ж
Dsequential_1/random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/random_flip_1/random_flip_up_down/strided_slice/stack_2ќ
<sequential_1/random_flip_1/random_flip_up_down/strided_sliceStridedSlice=sequential_1/random_flip_1/random_flip_up_down/Shape:output:0Ksequential_1/random_flip_1/random_flip_up_down/strided_slice/stack:output:0Msequential_1/random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0Msequential_1/random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_1/random_flip_1/random_flip_up_down/strided_slice
Csequential_1/random_flip_1/random_flip_up_down/random_uniform/shapePackEsequential_1/random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:2E
Csequential_1/random_flip_1/random_flip_up_down/random_uniform/shapeЫ
Asequential_1/random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2C
Asequential_1/random_flip_1/random_flip_up_down/random_uniform/minЫ
Asequential_1/random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2C
Asequential_1/random_flip_1/random_flip_up_down/random_uniform/maxД
Ksequential_1/random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniformLsequential_1/random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02M
Ksequential_1/random_flip_1/random_flip_up_down/random_uniform/RandomUniformн
Asequential_1/random_flip_1/random_flip_up_down/random_uniform/MulMulTsequential_1/random_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0Jsequential_1/random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2C
Asequential_1/random_flip_1/random_flip_up_down/random_uniform/MulТ
>sequential_1/random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/random_flip_1/random_flip_up_down/Reshape/shape/1Т
>sequential_1/random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/random_flip_1/random_flip_up_down/Reshape/shape/2Т
>sequential_1/random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/random_flip_1/random_flip_up_down/Reshape/shape/3д
<sequential_1/random_flip_1/random_flip_up_down/Reshape/shapePackEsequential_1/random_flip_1/random_flip_up_down/strided_slice:output:0Gsequential_1/random_flip_1/random_flip_up_down/Reshape/shape/1:output:0Gsequential_1/random_flip_1/random_flip_up_down/Reshape/shape/2:output:0Gsequential_1/random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2>
<sequential_1/random_flip_1/random_flip_up_down/Reshape/shapeУ
6sequential_1/random_flip_1/random_flip_up_down/ReshapeReshapeEsequential_1/random_flip_1/random_flip_up_down/random_uniform/Mul:z:0Esequential_1/random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ28
6sequential_1/random_flip_1/random_flip_up_down/Reshape№
4sequential_1/random_flip_1/random_flip_up_down/RoundRound?sequential_1/random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_flip_1/random_flip_up_down/RoundШ
=sequential_1/random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2?
=sequential_1/random_flip_1/random_flip_up_down/ReverseV2/axisб
8sequential_1/random_flip_1/random_flip_up_down/ReverseV2	ReverseV2Jsequential_1/random_flip_1/random_flip_up_down/control_dependency:output:0Fsequential_1/random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2:
8sequential_1/random_flip_1/random_flip_up_down/ReverseV2Ј
2sequential_1/random_flip_1/random_flip_up_down/mulMul8sequential_1/random_flip_1/random_flip_up_down/Round:y:0Asequential_1/random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ24
2sequential_1/random_flip_1/random_flip_up_down/mulБ
4sequential_1/random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_flip_1/random_flip_up_down/sub/xЂ
2sequential_1/random_flip_1/random_flip_up_down/subSub=sequential_1/random_flip_1/random_flip_up_down/sub/x:output:08sequential_1/random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ24
2sequential_1/random_flip_1/random_flip_up_down/subГ
4sequential_1/random_flip_1/random_flip_up_down/mul_1Mul6sequential_1/random_flip_1/random_flip_up_down/sub:z:0Jsequential_1/random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_flip_1/random_flip_up_down/mul_1
2sequential_1/random_flip_1/random_flip_up_down/addAddV26sequential_1/random_flip_1/random_flip_up_down/mul:z:08sequential_1/random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ24
2sequential_1/random_flip_1/random_flip_up_down/addВ
$sequential_1/random_rotation_1/ShapeShape6sequential_1/random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2&
$sequential_1/random_rotation_1/ShapeВ
2sequential_1/random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_1/random_rotation_1/strided_slice/stackЖ
4sequential_1/random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_1/random_rotation_1/strided_slice/stack_1Ж
4sequential_1/random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_1/random_rotation_1/strided_slice/stack_2
,sequential_1/random_rotation_1/strided_sliceStridedSlice-sequential_1/random_rotation_1/Shape:output:0;sequential_1/random_rotation_1/strided_slice/stack:output:0=sequential_1/random_rotation_1/strided_slice/stack_1:output:0=sequential_1/random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_1/random_rotation_1/strided_sliceЖ
4sequential_1/random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential_1/random_rotation_1/strided_slice_1/stackК
6sequential_1/random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_1/random_rotation_1/strided_slice_1/stack_1К
6sequential_1/random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_1/random_rotation_1/strided_slice_1/stack_2І
.sequential_1/random_rotation_1/strided_slice_1StridedSlice-sequential_1/random_rotation_1/Shape:output:0=sequential_1/random_rotation_1/strided_slice_1/stack:output:0?sequential_1/random_rotation_1/strided_slice_1/stack_1:output:0?sequential_1/random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_1/random_rotation_1/strided_slice_1Л
#sequential_1/random_rotation_1/CastCast7sequential_1/random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#sequential_1/random_rotation_1/CastЖ
4sequential_1/random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential_1/random_rotation_1/strided_slice_2/stackК
6sequential_1/random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_1/random_rotation_1/strided_slice_2/stack_1К
6sequential_1/random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_1/random_rotation_1/strided_slice_2/stack_2І
.sequential_1/random_rotation_1/strided_slice_2StridedSlice-sequential_1/random_rotation_1/Shape:output:0=sequential_1/random_rotation_1/strided_slice_2/stack:output:0?sequential_1/random_rotation_1/strided_slice_2/stack_1:output:0?sequential_1/random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_1/random_rotation_1/strided_slice_2П
%sequential_1/random_rotation_1/Cast_1Cast7sequential_1/random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%sequential_1/random_rotation_1/Cast_1л
5sequential_1/random_rotation_1/stateful_uniform/shapePack5sequential_1/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:27
5sequential_1/random_rotation_1/stateful_uniform/shapeЏ
3sequential_1/random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|й П25
3sequential_1/random_rotation_1/stateful_uniform/minЏ
3sequential_1/random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|й ?25
3sequential_1/random_rotation_1/stateful_uniform/maxи
Isequential_1/random_rotation_1/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2K
Isequential_1/random_rotation_1/stateful_uniform/StatefulUniform/algorithmЋ
?sequential_1/random_rotation_1/stateful_uniform/StatefulUniformStatefulUniformHsequential_1_random_rotation_1_stateful_uniform_statefuluniform_resourceRsequential_1/random_rotation_1/stateful_uniform/StatefulUniform/algorithm:output:0>sequential_1/random_rotation_1/stateful_uniform/shape:output:0*#
_output_shapes
:џџџџџџџџџ*
shape_dtype02A
?sequential_1/random_rotation_1/stateful_uniform/StatefulUniform
3sequential_1/random_rotation_1/stateful_uniform/subSub<sequential_1/random_rotation_1/stateful_uniform/max:output:0<sequential_1/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 25
3sequential_1/random_rotation_1/stateful_uniform/subЂ
3sequential_1/random_rotation_1/stateful_uniform/mulMulHsequential_1/random_rotation_1/stateful_uniform/StatefulUniform:output:07sequential_1/random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ25
3sequential_1/random_rotation_1/stateful_uniform/mul
/sequential_1/random_rotation_1/stateful_uniformAdd7sequential_1/random_rotation_1/stateful_uniform/mul:z:0<sequential_1/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:џџџџџџџџџ21
/sequential_1/random_rotation_1/stateful_uniformБ
4sequential_1/random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation_1/rotation_matrix/sub/yњ
2sequential_1/random_rotation_1/rotation_matrix/subSub)sequential_1/random_rotation_1/Cast_1:y:0=sequential_1/random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation_1/rotation_matrix/subв
2sequential_1/random_rotation_1/rotation_matrix/CosCos3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ24
2sequential_1/random_rotation_1/rotation_matrix/CosЕ
6sequential_1/random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6sequential_1/random_rotation_1/rotation_matrix/sub_1/y
4sequential_1/random_rotation_1/rotation_matrix/sub_1Sub)sequential_1/random_rotation_1/Cast_1:y:0?sequential_1/random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 26
4sequential_1/random_rotation_1/rotation_matrix/sub_1
2sequential_1/random_rotation_1/rotation_matrix/mulMul6sequential_1/random_rotation_1/rotation_matrix/Cos:y:08sequential_1/random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ24
2sequential_1/random_rotation_1/rotation_matrix/mulв
2sequential_1/random_rotation_1/rotation_matrix/SinSin3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ24
2sequential_1/random_rotation_1/rotation_matrix/SinЕ
6sequential_1/random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6sequential_1/random_rotation_1/rotation_matrix/sub_2/yў
4sequential_1/random_rotation_1/rotation_matrix/sub_2Sub'sequential_1/random_rotation_1/Cast:y:0?sequential_1/random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 26
4sequential_1/random_rotation_1/rotation_matrix/sub_2
4sequential_1/random_rotation_1/rotation_matrix/mul_1Mul6sequential_1/random_rotation_1/rotation_matrix/Sin:y:08sequential_1/random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/mul_1
4sequential_1/random_rotation_1/rotation_matrix/sub_3Sub6sequential_1/random_rotation_1/rotation_matrix/mul:z:08sequential_1/random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/sub_3
4sequential_1/random_rotation_1/rotation_matrix/sub_4Sub6sequential_1/random_rotation_1/rotation_matrix/sub:z:08sequential_1/random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/sub_4Й
8sequential_1/random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2:
8sequential_1/random_rotation_1/rotation_matrix/truediv/yІ
6sequential_1/random_rotation_1/rotation_matrix/truedivRealDiv8sequential_1/random_rotation_1/rotation_matrix/sub_4:z:0Asequential_1/random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ28
6sequential_1/random_rotation_1/rotation_matrix/truedivЕ
6sequential_1/random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6sequential_1/random_rotation_1/rotation_matrix/sub_5/yў
4sequential_1/random_rotation_1/rotation_matrix/sub_5Sub'sequential_1/random_rotation_1/Cast:y:0?sequential_1/random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 26
4sequential_1/random_rotation_1/rotation_matrix/sub_5ж
4sequential_1/random_rotation_1/rotation_matrix/Sin_1Sin3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/Sin_1Е
6sequential_1/random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6sequential_1/random_rotation_1/rotation_matrix/sub_6/y
4sequential_1/random_rotation_1/rotation_matrix/sub_6Sub)sequential_1/random_rotation_1/Cast_1:y:0?sequential_1/random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 26
4sequential_1/random_rotation_1/rotation_matrix/sub_6
4sequential_1/random_rotation_1/rotation_matrix/mul_2Mul8sequential_1/random_rotation_1/rotation_matrix/Sin_1:y:08sequential_1/random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/mul_2ж
4sequential_1/random_rotation_1/rotation_matrix/Cos_1Cos3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/Cos_1Е
6sequential_1/random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6sequential_1/random_rotation_1/rotation_matrix/sub_7/yў
4sequential_1/random_rotation_1/rotation_matrix/sub_7Sub'sequential_1/random_rotation_1/Cast:y:0?sequential_1/random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 26
4sequential_1/random_rotation_1/rotation_matrix/sub_7
4sequential_1/random_rotation_1/rotation_matrix/mul_3Mul8sequential_1/random_rotation_1/rotation_matrix/Cos_1:y:08sequential_1/random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/mul_3
2sequential_1/random_rotation_1/rotation_matrix/addAddV28sequential_1/random_rotation_1/rotation_matrix/mul_2:z:08sequential_1/random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ24
2sequential_1/random_rotation_1/rotation_matrix/add
4sequential_1/random_rotation_1/rotation_matrix/sub_8Sub8sequential_1/random_rotation_1/rotation_matrix/sub_5:z:06sequential_1/random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/sub_8Н
:sequential_1/random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:sequential_1/random_rotation_1/rotation_matrix/truediv_1/yЌ
8sequential_1/random_rotation_1/rotation_matrix/truediv_1RealDiv8sequential_1/random_rotation_1/rotation_matrix/sub_8:z:0Csequential_1/random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2:
8sequential_1/random_rotation_1/rotation_matrix/truediv_1Я
4sequential_1/random_rotation_1/rotation_matrix/ShapeShape3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:26
4sequential_1/random_rotation_1/rotation_matrix/Shapeв
Bsequential_1/random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_1/random_rotation_1/rotation_matrix/strided_slice/stackж
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice/stack_1ж
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice/stack_2ќ
<sequential_1/random_rotation_1/rotation_matrix/strided_sliceStridedSlice=sequential_1/random_rotation_1/rotation_matrix/Shape:output:0Ksequential_1/random_rotation_1/rotation_matrix/strided_slice/stack:output:0Msequential_1/random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0Msequential_1/random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_1/random_rotation_1/rotation_matrix/strided_sliceж
4sequential_1/random_rotation_1/rotation_matrix/Cos_2Cos3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/Cos_2н
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stackс
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stack_1с
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stack_2Б
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_1StridedSlice8sequential_1/random_rotation_1/rotation_matrix/Cos_2:y:0Msequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_1ж
4sequential_1/random_rotation_1/rotation_matrix/Sin_2Sin3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/Sin_2н
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stackс
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stack_1с
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stack_2Б
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_2StridedSlice8sequential_1/random_rotation_1/rotation_matrix/Sin_2:y:0Msequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_2ъ
2sequential_1/random_rotation_1/rotation_matrix/NegNegGsequential_1/random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ24
2sequential_1/random_rotation_1/rotation_matrix/Negн
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stackс
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stack_1с
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stack_2Г
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_3StridedSlice:sequential_1/random_rotation_1/rotation_matrix/truediv:z:0Msequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_3ж
4sequential_1/random_rotation_1/rotation_matrix/Sin_3Sin3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/Sin_3н
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stackс
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stack_1с
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stack_2Б
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_4StridedSlice8sequential_1/random_rotation_1/rotation_matrix/Sin_3:y:0Msequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_4ж
4sequential_1/random_rotation_1/rotation_matrix/Cos_3Cos3sequential_1/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/Cos_3н
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stackс
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stack_1с
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stack_2Б
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_5StridedSlice8sequential_1/random_rotation_1/rotation_matrix/Cos_3:y:0Msequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_5н
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stackс
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stack_1с
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stack_2Е
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_6StridedSlice<sequential_1/random_rotation_1/rotation_matrix/truediv_1:z:0Msequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Osequential_1/random_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_1/random_rotation_1/rotation_matrix/strided_slice_6К
:sequential_1/random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_1/random_rotation_1/rotation_matrix/zeros/mul/yЈ
8sequential_1/random_rotation_1/rotation_matrix/zeros/mulMulEsequential_1/random_rotation_1/rotation_matrix/strided_slice:output:0Csequential_1/random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_1/random_rotation_1/rotation_matrix/zeros/mulН
;sequential_1/random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2=
;sequential_1/random_rotation_1/rotation_matrix/zeros/Less/yЃ
9sequential_1/random_rotation_1/rotation_matrix/zeros/LessLess<sequential_1/random_rotation_1/rotation_matrix/zeros/mul:z:0Dsequential_1/random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_1/random_rotation_1/rotation_matrix/zeros/LessР
=sequential_1/random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_1/random_rotation_1/rotation_matrix/zeros/packed/1П
;sequential_1/random_rotation_1/rotation_matrix/zeros/packedPackEsequential_1/random_rotation_1/rotation_matrix/strided_slice:output:0Fsequential_1/random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_1/random_rotation_1/rotation_matrix/zeros/packedН
:sequential_1/random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_1/random_rotation_1/rotation_matrix/zeros/ConstБ
4sequential_1/random_rotation_1/rotation_matrix/zerosFillDsequential_1/random_rotation_1/rotation_matrix/zeros/packed:output:0Csequential_1/random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ26
4sequential_1/random_rotation_1/rotation_matrix/zerosК
:sequential_1/random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_1/random_rotation_1/rotation_matrix/concat/axisо
5sequential_1/random_rotation_1/rotation_matrix/concatConcatV2Gsequential_1/random_rotation_1/rotation_matrix/strided_slice_1:output:06sequential_1/random_rotation_1/rotation_matrix/Neg:y:0Gsequential_1/random_rotation_1/rotation_matrix/strided_slice_3:output:0Gsequential_1/random_rotation_1/rotation_matrix/strided_slice_4:output:0Gsequential_1/random_rotation_1/rotation_matrix/strided_slice_5:output:0Gsequential_1/random_rotation_1/rotation_matrix/strided_slice_6:output:0=sequential_1/random_rotation_1/rotation_matrix/zeros:output:0Csequential_1/random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ27
5sequential_1/random_rotation_1/rotation_matrix/concatЦ
.sequential_1/random_rotation_1/transform/ShapeShape6sequential_1/random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:20
.sequential_1/random_rotation_1/transform/ShapeЦ
<sequential_1/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_1/random_rotation_1/transform/strided_slice/stackЪ
>sequential_1/random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_1/random_rotation_1/transform/strided_slice/stack_1Ъ
>sequential_1/random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_1/random_rotation_1/transform/strided_slice/stack_2Ф
6sequential_1/random_rotation_1/transform/strided_sliceStridedSlice7sequential_1/random_rotation_1/transform/Shape:output:0Esequential_1/random_rotation_1/transform/strided_slice/stack:output:0Gsequential_1/random_rotation_1/transform/strided_slice/stack_1:output:0Gsequential_1/random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:28
6sequential_1/random_rotation_1/transform/strided_sliceЏ
3sequential_1/random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3sequential_1/random_rotation_1/transform/fill_value
Csequential_1/random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV36sequential_1/random_flip_1/random_flip_up_down/add:z:0>sequential_1/random_rotation_1/rotation_matrix/concat:output:0?sequential_1/random_rotation_1/transform/strided_slice:output:0<sequential_1/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2E
Csequential_1/random_rotation_1/transform/ImageProjectiveTransformV3Ь
 sequential_1/random_zoom_1/ShapeShapeXsequential_1/random_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2"
 sequential_1/random_zoom_1/ShapeЊ
.sequential_1/random_zoom_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_1/random_zoom_1/strided_slice/stackЎ
0sequential_1/random_zoom_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_1/random_zoom_1/strided_slice/stack_1Ў
0sequential_1/random_zoom_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_1/random_zoom_1/strided_slice/stack_2
(sequential_1/random_zoom_1/strided_sliceStridedSlice)sequential_1/random_zoom_1/Shape:output:07sequential_1/random_zoom_1/strided_slice/stack:output:09sequential_1/random_zoom_1/strided_slice/stack_1:output:09sequential_1/random_zoom_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_1/random_zoom_1/strided_sliceЎ
0sequential_1/random_zoom_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_1/random_zoom_1/strided_slice_1/stackВ
2sequential_1/random_zoom_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/random_zoom_1/strided_slice_1/stack_1В
2sequential_1/random_zoom_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/random_zoom_1/strided_slice_1/stack_2
*sequential_1/random_zoom_1/strided_slice_1StridedSlice)sequential_1/random_zoom_1/Shape:output:09sequential_1/random_zoom_1/strided_slice_1/stack:output:0;sequential_1/random_zoom_1/strided_slice_1/stack_1:output:0;sequential_1/random_zoom_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_1/random_zoom_1/strided_slice_1Џ
sequential_1/random_zoom_1/CastCast3sequential_1/random_zoom_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2!
sequential_1/random_zoom_1/CastЎ
0sequential_1/random_zoom_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_1/random_zoom_1/strided_slice_2/stackВ
2sequential_1/random_zoom_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/random_zoom_1/strided_slice_2/stack_1В
2sequential_1/random_zoom_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/random_zoom_1/strided_slice_2/stack_2
*sequential_1/random_zoom_1/strided_slice_2StridedSlice)sequential_1/random_zoom_1/Shape:output:09sequential_1/random_zoom_1/strided_slice_2/stack:output:0;sequential_1/random_zoom_1/strided_slice_2/stack_1:output:0;sequential_1/random_zoom_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_1/random_zoom_1/strided_slice_2Г
!sequential_1/random_zoom_1/Cast_1Cast3sequential_1/random_zoom_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!sequential_1/random_zoom_1/Cast_1Ќ
3sequential_1/random_zoom_1/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_1/random_zoom_1/stateful_uniform/shape/1
1sequential_1/random_zoom_1/stateful_uniform/shapePack1sequential_1/random_zoom_1/strided_slice:output:0<sequential_1/random_zoom_1/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:23
1sequential_1/random_zoom_1/stateful_uniform/shapeЇ
/sequential_1/random_zoom_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?21
/sequential_1/random_zoom_1/stateful_uniform/minЇ
/sequential_1/random_zoom_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?21
/sequential_1/random_zoom_1/stateful_uniform/maxа
Esequential_1/random_zoom_1/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2G
Esequential_1/random_zoom_1/stateful_uniform/StatefulUniform/algorithm
;sequential_1/random_zoom_1/stateful_uniform/StatefulUniformStatefulUniformDsequential_1_random_zoom_1_stateful_uniform_statefuluniform_resourceNsequential_1/random_zoom_1/stateful_uniform/StatefulUniform/algorithm:output:0:sequential_1/random_zoom_1/stateful_uniform/shape:output:0*'
_output_shapes
:џџџџџџџџџ*
shape_dtype02=
;sequential_1/random_zoom_1/stateful_uniform/StatefulUniformў
/sequential_1/random_zoom_1/stateful_uniform/subSub8sequential_1/random_zoom_1/stateful_uniform/max:output:08sequential_1/random_zoom_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 21
/sequential_1/random_zoom_1/stateful_uniform/sub
/sequential_1/random_zoom_1/stateful_uniform/mulMulDsequential_1/random_zoom_1/stateful_uniform/StatefulUniform:output:03sequential_1/random_zoom_1/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ21
/sequential_1/random_zoom_1/stateful_uniform/mul
+sequential_1/random_zoom_1/stateful_uniformAdd3sequential_1/random_zoom_1/stateful_uniform/mul:z:08sequential_1/random_zoom_1/stateful_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2-
+sequential_1/random_zoom_1/stateful_uniform
&sequential_1/random_zoom_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_1/random_zoom_1/concat/axis 
!sequential_1/random_zoom_1/concatConcatV2/sequential_1/random_zoom_1/stateful_uniform:z:0/sequential_1/random_zoom_1/stateful_uniform:z:0/sequential_1/random_zoom_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2#
!sequential_1/random_zoom_1/concatЖ
,sequential_1/random_zoom_1/zoom_matrix/ShapeShape*sequential_1/random_zoom_1/concat:output:0*
T0*
_output_shapes
:2.
,sequential_1/random_zoom_1/zoom_matrix/ShapeТ
:sequential_1/random_zoom_1/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sequential_1/random_zoom_1/zoom_matrix/strided_slice/stackЦ
<sequential_1/random_zoom_1/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_1/random_zoom_1/zoom_matrix/strided_slice/stack_1Ц
<sequential_1/random_zoom_1/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_1/random_zoom_1/zoom_matrix/strided_slice/stack_2Ь
4sequential_1/random_zoom_1/zoom_matrix/strided_sliceStridedSlice5sequential_1/random_zoom_1/zoom_matrix/Shape:output:0Csequential_1/random_zoom_1/zoom_matrix/strided_slice/stack:output:0Esequential_1/random_zoom_1/zoom_matrix/strided_slice/stack_1:output:0Esequential_1/random_zoom_1/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4sequential_1/random_zoom_1/zoom_matrix/strided_sliceЁ
,sequential_1/random_zoom_1/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,sequential_1/random_zoom_1/zoom_matrix/sub/yо
*sequential_1/random_zoom_1/zoom_matrix/subSub%sequential_1/random_zoom_1/Cast_1:y:05sequential_1/random_zoom_1/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2,
*sequential_1/random_zoom_1/zoom_matrix/subЉ
0sequential_1/random_zoom_1/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @22
0sequential_1/random_zoom_1/zoom_matrix/truediv/yї
.sequential_1/random_zoom_1/zoom_matrix/truedivRealDiv.sequential_1/random_zoom_1/zoom_matrix/sub:z:09sequential_1/random_zoom_1/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 20
.sequential_1/random_zoom_1/zoom_matrix/truedivб
<sequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2>
<sequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stackе
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2@
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stack_1е
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2@
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stack_2
6sequential_1/random_zoom_1/zoom_matrix/strided_slice_1StridedSlice*sequential_1/random_zoom_1/concat:output:0Esequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stack:output:0Gsequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stack_1:output:0Gsequential_1/random_zoom_1/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask28
6sequential_1/random_zoom_1/zoom_matrix/strided_slice_1Ѕ
.sequential_1/random_zoom_1/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.sequential_1/random_zoom_1/zoom_matrix/sub_1/x
,sequential_1/random_zoom_1/zoom_matrix/sub_1Sub7sequential_1/random_zoom_1/zoom_matrix/sub_1/x:output:0?sequential_1/random_zoom_1/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,sequential_1/random_zoom_1/zoom_matrix/sub_1ї
*sequential_1/random_zoom_1/zoom_matrix/mulMul2sequential_1/random_zoom_1/zoom_matrix/truediv:z:00sequential_1/random_zoom_1/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*sequential_1/random_zoom_1/zoom_matrix/mulЅ
.sequential_1/random_zoom_1/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.sequential_1/random_zoom_1/zoom_matrix/sub_2/yт
,sequential_1/random_zoom_1/zoom_matrix/sub_2Sub#sequential_1/random_zoom_1/Cast:y:07sequential_1/random_zoom_1/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2.
,sequential_1/random_zoom_1/zoom_matrix/sub_2­
2sequential_1/random_zoom_1/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @24
2sequential_1/random_zoom_1/zoom_matrix/truediv_1/yџ
0sequential_1/random_zoom_1/zoom_matrix/truediv_1RealDiv0sequential_1/random_zoom_1/zoom_matrix/sub_2:z:0;sequential_1/random_zoom_1/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 22
0sequential_1/random_zoom_1/zoom_matrix/truediv_1б
<sequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2>
<sequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stackе
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2@
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stack_1е
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2@
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stack_2
6sequential_1/random_zoom_1/zoom_matrix/strided_slice_2StridedSlice*sequential_1/random_zoom_1/concat:output:0Esequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stack:output:0Gsequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stack_1:output:0Gsequential_1/random_zoom_1/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask28
6sequential_1/random_zoom_1/zoom_matrix/strided_slice_2Ѕ
.sequential_1/random_zoom_1/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.sequential_1/random_zoom_1/zoom_matrix/sub_3/x
,sequential_1/random_zoom_1/zoom_matrix/sub_3Sub7sequential_1/random_zoom_1/zoom_matrix/sub_3/x:output:0?sequential_1/random_zoom_1/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,sequential_1/random_zoom_1/zoom_matrix/sub_3§
,sequential_1/random_zoom_1/zoom_matrix/mul_1Mul4sequential_1/random_zoom_1/zoom_matrix/truediv_1:z:00sequential_1/random_zoom_1/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,sequential_1/random_zoom_1/zoom_matrix/mul_1б
<sequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2>
<sequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stackе
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2@
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stack_1е
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2@
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stack_2
6sequential_1/random_zoom_1/zoom_matrix/strided_slice_3StridedSlice*sequential_1/random_zoom_1/concat:output:0Esequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stack:output:0Gsequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stack_1:output:0Gsequential_1/random_zoom_1/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask28
6sequential_1/random_zoom_1/zoom_matrix/strided_slice_3Њ
2sequential_1/random_zoom_1/zoom_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_1/random_zoom_1/zoom_matrix/zeros/mul/y
0sequential_1/random_zoom_1/zoom_matrix/zeros/mulMul=sequential_1/random_zoom_1/zoom_matrix/strided_slice:output:0;sequential_1/random_zoom_1/zoom_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential_1/random_zoom_1/zoom_matrix/zeros/mul­
3sequential_1/random_zoom_1/zoom_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш25
3sequential_1/random_zoom_1/zoom_matrix/zeros/Less/y
1sequential_1/random_zoom_1/zoom_matrix/zeros/LessLess4sequential_1/random_zoom_1/zoom_matrix/zeros/mul:z:0<sequential_1/random_zoom_1/zoom_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 23
1sequential_1/random_zoom_1/zoom_matrix/zeros/LessА
5sequential_1/random_zoom_1/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/random_zoom_1/zoom_matrix/zeros/packed/1
3sequential_1/random_zoom_1/zoom_matrix/zeros/packedPack=sequential_1/random_zoom_1/zoom_matrix/strided_slice:output:0>sequential_1/random_zoom_1/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:25
3sequential_1/random_zoom_1/zoom_matrix/zeros/packed­
2sequential_1/random_zoom_1/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    24
2sequential_1/random_zoom_1/zoom_matrix/zeros/Const
,sequential_1/random_zoom_1/zoom_matrix/zerosFill<sequential_1/random_zoom_1/zoom_matrix/zeros/packed:output:0;sequential_1/random_zoom_1/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,sequential_1/random_zoom_1/zoom_matrix/zerosЎ
4sequential_1/random_zoom_1/zoom_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_1/random_zoom_1/zoom_matrix/zeros_1/mul/y
2sequential_1/random_zoom_1/zoom_matrix/zeros_1/mulMul=sequential_1/random_zoom_1/zoom_matrix/strided_slice:output:0=sequential_1/random_zoom_1/zoom_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_zoom_1/zoom_matrix/zeros_1/mulБ
5sequential_1/random_zoom_1/zoom_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш27
5sequential_1/random_zoom_1/zoom_matrix/zeros_1/Less/y
3sequential_1/random_zoom_1/zoom_matrix/zeros_1/LessLess6sequential_1/random_zoom_1/zoom_matrix/zeros_1/mul:z:0>sequential_1/random_zoom_1/zoom_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 25
3sequential_1/random_zoom_1/zoom_matrix/zeros_1/LessД
7sequential_1/random_zoom_1/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :29
7sequential_1/random_zoom_1/zoom_matrix/zeros_1/packed/1Ѕ
5sequential_1/random_zoom_1/zoom_matrix/zeros_1/packedPack=sequential_1/random_zoom_1/zoom_matrix/strided_slice:output:0@sequential_1/random_zoom_1/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:27
5sequential_1/random_zoom_1/zoom_matrix/zeros_1/packedБ
4sequential_1/random_zoom_1/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4sequential_1/random_zoom_1/zoom_matrix/zeros_1/Const
.sequential_1/random_zoom_1/zoom_matrix/zeros_1Fill>sequential_1/random_zoom_1/zoom_matrix/zeros_1/packed:output:0=sequential_1/random_zoom_1/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ20
.sequential_1/random_zoom_1/zoom_matrix/zeros_1б
<sequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2>
<sequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stackе
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2@
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stack_1е
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2@
>sequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stack_2
6sequential_1/random_zoom_1/zoom_matrix/strided_slice_4StridedSlice*sequential_1/random_zoom_1/concat:output:0Esequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stack:output:0Gsequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stack_1:output:0Gsequential_1/random_zoom_1/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask28
6sequential_1/random_zoom_1/zoom_matrix/strided_slice_4Ў
4sequential_1/random_zoom_1/zoom_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_1/random_zoom_1/zoom_matrix/zeros_2/mul/y
2sequential_1/random_zoom_1/zoom_matrix/zeros_2/mulMul=sequential_1/random_zoom_1/zoom_matrix/strided_slice:output:0=sequential_1/random_zoom_1/zoom_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_zoom_1/zoom_matrix/zeros_2/mulБ
5sequential_1/random_zoom_1/zoom_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш27
5sequential_1/random_zoom_1/zoom_matrix/zeros_2/Less/y
3sequential_1/random_zoom_1/zoom_matrix/zeros_2/LessLess6sequential_1/random_zoom_1/zoom_matrix/zeros_2/mul:z:0>sequential_1/random_zoom_1/zoom_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 25
3sequential_1/random_zoom_1/zoom_matrix/zeros_2/LessД
7sequential_1/random_zoom_1/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :29
7sequential_1/random_zoom_1/zoom_matrix/zeros_2/packed/1Ѕ
5sequential_1/random_zoom_1/zoom_matrix/zeros_2/packedPack=sequential_1/random_zoom_1/zoom_matrix/strided_slice:output:0@sequential_1/random_zoom_1/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:27
5sequential_1/random_zoom_1/zoom_matrix/zeros_2/packedБ
4sequential_1/random_zoom_1/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4sequential_1/random_zoom_1/zoom_matrix/zeros_2/Const
.sequential_1/random_zoom_1/zoom_matrix/zeros_2Fill>sequential_1/random_zoom_1/zoom_matrix/zeros_2/packed:output:0=sequential_1/random_zoom_1/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ20
.sequential_1/random_zoom_1/zoom_matrix/zeros_2Њ
2sequential_1/random_zoom_1/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_1/random_zoom_1/zoom_matrix/concat/axisя
-sequential_1/random_zoom_1/zoom_matrix/concatConcatV2?sequential_1/random_zoom_1/zoom_matrix/strided_slice_3:output:05sequential_1/random_zoom_1/zoom_matrix/zeros:output:0.sequential_1/random_zoom_1/zoom_matrix/mul:z:07sequential_1/random_zoom_1/zoom_matrix/zeros_1:output:0?sequential_1/random_zoom_1/zoom_matrix/strided_slice_4:output:00sequential_1/random_zoom_1/zoom_matrix/mul_1:z:07sequential_1/random_zoom_1/zoom_matrix/zeros_2:output:0;sequential_1/random_zoom_1/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2/
-sequential_1/random_zoom_1/zoom_matrix/concatр
*sequential_1/random_zoom_1/transform/ShapeShapeXsequential_1/random_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2,
*sequential_1/random_zoom_1/transform/ShapeО
8sequential_1/random_zoom_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_1/random_zoom_1/transform/strided_slice/stackТ
:sequential_1/random_zoom_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/random_zoom_1/transform/strided_slice/stack_1Т
:sequential_1/random_zoom_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/random_zoom_1/transform/strided_slice/stack_2Ќ
2sequential_1/random_zoom_1/transform/strided_sliceStridedSlice3sequential_1/random_zoom_1/transform/Shape:output:0Asequential_1/random_zoom_1/transform/strided_slice/stack:output:0Csequential_1/random_zoom_1/transform/strided_slice/stack_1:output:0Csequential_1/random_zoom_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:24
2sequential_1/random_zoom_1/transform/strided_sliceЇ
/sequential_1/random_zoom_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential_1/random_zoom_1/transform/fill_value
?sequential_1/random_zoom_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Xsequential_1/random_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:06sequential_1/random_zoom_1/zoom_matrix/concat:output:0;sequential_1/random_zoom_1/transform/strided_slice:output:08sequential_1/random_zoom_1/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2A
?sequential_1/random_zoom_1/transform/ImageProjectiveTransformV3i
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/xв
rescaling/mulMulTsequential_1/random_zoom_1/transform/ImageProjectiveTransformV3:transformed_images:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/addЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpЦ
conv2d/Conv2DConv2Drescaling/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@*
paddingVALID*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpІ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2
conv2d/BiasAdd
activation/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2
activation/ReluХ
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@*
paddingVALID*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2
conv2d_1/BiasAdd
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2
activation_1/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpй
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_2/BiasAdd
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
activation_2/ReluЫ
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolА
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpй
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_3/Conv2DЇ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpЌ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_3/BiasAdd
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
activation_3/ReluЫ
max_pooling2d_3/MaxPoolMaxPoolactivation_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolА
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOpй
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_4/Conv2DЇ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpЌ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_4/BiasAdd
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
activation_4/ReluЫ
max_pooling2d_4/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/dropout/Const­
dropout/dropout/MulMul max_pooling2d_4/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/dropout/Mul~
dropout/dropout/ShapeShape max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeд
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2 
dropout/dropout/GreaterEqual/yц
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ@2
dropout/dropout/CastЂ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 	  2
flatten/Const
flatten/ReshapeReshapedropout/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten/ReshapeЁ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

dense/ReluІ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_1/SoftmaxЖ
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp@^sequential_1/random_rotation_1/stateful_uniform/StatefulUniform<^sequential_1/random_zoom_1/stateful_uniform/StatefulUniform*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2
?sequential_1/random_rotation_1/stateful_uniform/StatefulUniform?sequential_1/random_rotation_1/stateful_uniform/StatefulUniform2z
;sequential_1/random_zoom_1/stateful_uniform/StatefulUniform;sequential_1/random_zoom_1/stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь	
о
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1723341

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ}}@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


И
,__inference_sequential_layer_call_fn_1724329

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*0
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_17238022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЎS
§
G__inference_sequential_layer_call_and_return_conditional_losses_1723802

inputs
conv2d_1723754
conv2d_1723756
conv2d_1_1723761
conv2d_1_1723763
conv2d_2_1723768
conv2d_2_1723770
conv2d_3_1723775
conv2d_3_1723777
conv2d_4_1723782
conv2d_4_1723784
dense_1723791
dense_1723793
dense_1_1723796
dense_1_1723798
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallі
sequential_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_17232012
sequential_1/PartitionedCalli
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/xЃ
rescaling/mulMul%sequential_1/PartitionedCall:output:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/add­
conv2d/StatefulPartitionedCallStatefulPartitionedCallrescaling/add:z:0conv2d_1723754conv2d_1723756*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_17233012 
conv2d/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_17233222
activation/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_17232102
max_pooling2d/PartitionedCallЪ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1723761conv2d_1_1723763*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_17233412"
 conv2d_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_17233622
activation_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>>@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17232222!
max_pooling2d_1/PartitionedCallЬ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1723768conv2d_2_1723770*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_17233812"
 conv2d_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_17234022
activation_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17232342!
max_pooling2d_2/PartitionedCallЬ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1723775conv2d_3_1723777*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_17234212"
 conv2d_3/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_17234422
activation_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17232462!
max_pooling2d_3/PartitionedCallЬ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_1723782conv2d_4_1723784*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_17234612"
 conv2d_4/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_17234822
activation_4/PartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17232582!
max_pooling2d_4/PartitionedCall
dropout/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_17235082
dropout/PartitionedCallј
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17235272
flatten/PartitionedCallЎ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1723791dense_1723793*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17235462
dense/StatefulPartitionedCallН
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1723796dense_1_1723798*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_17235732!
dense_1/StatefulPartitionedCallы
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџ::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
~
)__inference_dense_1_layer_call_fn_1724820

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_17235732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2d_1_layer_call_fn_1724645

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_17233412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ}}@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
 
r
I__inference_sequential_1_layer_call_and_return_conditional_losses_1722935
random_flip_1_input
identityq
IdentityIdentityrandom_flip_1_input*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџ:f b
1
_output_shapes
:џџџџџџџџџ
-
_user_specified_namerandom_flip_1_input
g
Л
"__inference__wrapped_model_1722680
sequential_1_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource6
2sequential_conv2d_3_conv2d_readvariableop_resource7
3sequential_conv2d_3_biasadd_readvariableop_resource6
2sequential_conv2d_4_conv2d_readvariableop_resource7
3sequential_conv2d_4_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identityЂ(sequential/conv2d/BiasAdd/ReadVariableOpЂ'sequential/conv2d/Conv2D/ReadVariableOpЂ*sequential/conv2d_1/BiasAdd/ReadVariableOpЂ)sequential/conv2d_1/Conv2D/ReadVariableOpЂ*sequential/conv2d_2/BiasAdd/ReadVariableOpЂ)sequential/conv2d_2/Conv2D/ReadVariableOpЂ*sequential/conv2d_3/BiasAdd/ReadVariableOpЂ)sequential/conv2d_3/Conv2D/ReadVariableOpЂ*sequential/conv2d_4/BiasAdd/ReadVariableOpЂ)sequential/conv2d_4/Conv2D/ReadVariableOpЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOp
sequential/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
sequential/rescaling/Cast/x
sequential/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/rescaling/Cast_1/xБ
sequential/rescaling/mulMulsequential_1_input$sequential/rescaling/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
sequential/rescaling/mulП
sequential/rescaling/addAddV2sequential/rescaling/mul:z:0&sequential/rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
sequential/rescaling/addЫ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpђ
sequential/conv2d/Conv2DConv2Dsequential/rescaling/add:z:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@*
paddingVALID*
strides
2
sequential/conv2d/Conv2DТ
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOpв
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2
sequential/conv2d/BiasAdd 
sequential/activation/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2
sequential/activation/Reluц
 sequential/max_pooling2d/MaxPoolMaxPool(sequential/activation/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolб
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2DШ
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpи
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2
sequential/conv2d_1/BiasAddЄ
sequential/activation_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2
sequential/activation_1/Reluь
"sequential/max_pooling2d_1/MaxPoolMaxPool*sequential/activation_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ>>@*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPoolб
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
sequential/conv2d_2/Conv2DШ
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOpи
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
sequential/conv2d_2/BiasAddЄ
sequential/activation_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
sequential/activation_2/Reluь
"sequential/max_pooling2d_2/MaxPoolMaxPool*sequential/activation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPoolб
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_3/Conv2D/ReadVariableOp
sequential/conv2d_3/Conv2DConv2D+sequential/max_pooling2d_2/MaxPool:output:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
sequential/conv2d_3/Conv2DШ
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_3/BiasAdd/ReadVariableOpи
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
sequential/conv2d_3/BiasAddЄ
sequential/activation_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
sequential/activation_3/Reluь
"sequential/max_pooling2d_3/MaxPoolMaxPool*sequential/activation_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_3/MaxPoolб
)sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_4/Conv2D/ReadVariableOp
sequential/conv2d_4/Conv2DConv2D+sequential/max_pooling2d_3/MaxPool:output:01sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
sequential/conv2d_4/Conv2DШ
*sequential/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_4/BiasAdd/ReadVariableOpи
sequential/conv2d_4/BiasAddBiasAdd#sequential/conv2d_4/Conv2D:output:02sequential/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
sequential/conv2d_4/BiasAddЄ
sequential/activation_4/ReluRelu$sequential/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
sequential/activation_4/Reluь
"sequential/max_pooling2d_4/MaxPoolMaxPool*sequential/activation_4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_4/MaxPool­
sequential/dropout/IdentityIdentity+sequential/max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
sequential/dropout/Identity
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 	  2
sequential/flatten/ConstП
sequential/flatten/ReshapeReshape$sequential/dropout/Identity:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential/flatten/ReshapeТ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02(
&sequential/dense/MatMul/ReadVariableOpФ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential/dense/MatMulР
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpЦ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential/dense/BiasAdd
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential/dense/ReluЧ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpЩ
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
sequential/dense_1/MatMulХ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЭ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
sequential/dense_1/BiasAdd
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
sequential/dense_1/Softmaxл
IdentityIdentity$sequential/dense_1/Softmax:softmax:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp+^sequential/conv2d_4/BiasAdd/ReadVariableOp*^sequential/conv2d_4/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџ::::::::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2X
*sequential/conv2d_4/BiasAdd/ReadVariableOp*sequential/conv2d_4/BiasAdd/ReadVariableOp2V
)sequential/conv2d_4/Conv2D/ReadVariableOp)sequential/conv2d_4/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesequential_1_input
з
e
I__inference_activation_4_layer_call_and_return_conditional_losses_1723482

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
љ
e
I__inference_sequential_1_layer_call_and_return_conditional_losses_1723201

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2d_2_layer_call_fn_1724674

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_17233812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ>>@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ>>@
 
_user_specified_nameinputs
Ч
J
.__inference_activation_2_layer_call_fn_1724684

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_17234022
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<<@:W S
/
_output_shapes
:џџџџџџџџџ<<@
 
_user_specified_nameinputs
ѕ	
л
B__inference_dense_layer_call_and_return_conditional_losses_1724791

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
E
)__inference_flatten_layer_call_fn_1724780

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17235272
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ю

ф
,__inference_sequential_layer_call_fn_1723744
sequential_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallsequential_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*0
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_17237092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesequential_1_input
з
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1724679

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ<<@:W S
/
_output_shapes
:џџџџџџџџџ<<@
 
_user_specified_nameinputs
є	
м
C__inference_conv2d_layer_call_and_return_conditional_losses_1723301

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџўў@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1723222

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2d_4_layer_call_fn_1724732

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_17234612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
К
M
1__inference_max_pooling2d_4_layer_call_fn_1723264

inputs
identityј
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17232582
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


.__inference_sequential_1_layer_call_fn_1724592

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_17231882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь	
о
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1724636

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ}}@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ч
b
D__inference_dropout_layer_call_and_return_conditional_losses_1723508

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ь	
о
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1724665

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ>>@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ>>@
 
_user_specified_nameinputs
Ч
J
.__inference_activation_4_layer_call_fn_1724742

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_17234822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


*__inference_conv2d_3_layer_call_fn_1724703

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_17234212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
љЋ
й
I__inference_sequential_1_layer_call_and_return_conditional_losses_1722931
random_flip_1_input?
;random_rotation_1_stateful_uniform_statefuluniform_resource;
7random_zoom_1_stateful_uniform_statefuluniform_resource
identityЂ2random_rotation_1/stateful_uniform/StatefulUniformЂ.random_zoom_1/stateful_uniform/StatefulUniformї
7random_flip_1/random_flip_left_right/control_dependencyIdentityrandom_flip_1_input*
T0*&
_class
loc:@random_flip_1_input*1
_output_shapes
:џџџџџџџџџ29
7random_flip_1/random_flip_left_right/control_dependencyШ
*random_flip_1/random_flip_left_right/ShapeShape@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2,
*random_flip_1/random_flip_left_right/ShapeО
8random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_1/random_flip_left_right/strided_slice/stackТ
:random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_1Т
:random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_2Р
2random_flip_1/random_flip_left_right/strided_sliceStridedSlice3random_flip_1/random_flip_left_right/Shape:output:0Arandom_flip_1/random_flip_left_right/strided_slice/stack:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2random_flip_1/random_flip_left_right/strided_sliceщ
9random_flip_1/random_flip_left_right/random_uniform/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2;
9random_flip_1/random_flip_left_right/random_uniform/shapeЗ
7random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7random_flip_1/random_flip_left_right/random_uniform/minЗ
7random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7random_flip_1/random_flip_left_right/random_uniform/max
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformBrandom_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02C
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformЕ
7random_flip_1/random_flip_left_right/random_uniform/MulMulJrandom_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0@random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ29
7random_flip_1/random_flip_left_right/random_uniform/MulЎ
4random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/1Ў
4random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/2Ў
4random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/3
2random_flip_1/random_flip_left_right/Reshape/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0=random_flip_1/random_flip_left_right/Reshape/shape/1:output:0=random_flip_1/random_flip_left_right/Reshape/shape/2:output:0=random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:24
2random_flip_1/random_flip_left_right/Reshape/shape
,random_flip_1/random_flip_left_right/ReshapeReshape;random_flip_1/random_flip_left_right/random_uniform/Mul:z:0;random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2.
,random_flip_1/random_flip_left_right/Reshapeв
*random_flip_1/random_flip_left_right/RoundRound5random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2,
*random_flip_1/random_flip_left_right/RoundД
3random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:25
3random_flip_1/random_flip_left_right/ReverseV2/axisЉ
.random_flip_1/random_flip_left_right/ReverseV2	ReverseV2@random_flip_1/random_flip_left_right/control_dependency:output:0<random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ20
.random_flip_1/random_flip_left_right/ReverseV2
(random_flip_1/random_flip_left_right/mulMul.random_flip_1/random_flip_left_right/Round:y:07random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/mul
*random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_flip_1/random_flip_left_right/sub/xњ
(random_flip_1/random_flip_left_right/subSub3random_flip_1/random_flip_left_right/sub/x:output:0.random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/sub
*random_flip_1/random_flip_left_right/mul_1Mul,random_flip_1/random_flip_left_right/sub:z:0@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2,
*random_flip_1/random_flip_left_right/mul_1ї
(random_flip_1/random_flip_left_right/addAddV2,random_flip_1/random_flip_left_right/mul:z:0.random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/add
4random_flip_1/random_flip_up_down/control_dependencyIdentity,random_flip_1/random_flip_left_right/add:z:0*
T0*;
_class1
/-loc:@random_flip_1/random_flip_left_right/add*1
_output_shapes
:џџџџџџџџџ26
4random_flip_1/random_flip_up_down/control_dependencyП
'random_flip_1/random_flip_up_down/ShapeShape=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2)
'random_flip_1/random_flip_up_down/ShapeИ
5random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_flip_1/random_flip_up_down/strided_slice/stackМ
7random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_1М
7random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_2Ў
/random_flip_1/random_flip_up_down/strided_sliceStridedSlice0random_flip_1/random_flip_up_down/Shape:output:0>random_flip_1/random_flip_up_down/strided_slice/stack:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_flip_1/random_flip_up_down/strided_sliceр
6random_flip_1/random_flip_up_down/random_uniform/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6random_flip_1/random_flip_up_down/random_uniform/shapeБ
4random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4random_flip_1/random_flip_up_down/random_uniform/minБ
4random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4random_flip_1/random_flip_up_down/random_uniform/max
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniform?random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02@
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformЉ
4random_flip_1/random_flip_up_down/random_uniform/MulMulGrandom_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0=random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4random_flip_1/random_flip_up_down/random_uniform/MulЈ
1random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/1Ј
1random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/2Ј
1random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/3
/random_flip_1/random_flip_up_down/Reshape/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0:random_flip_1/random_flip_up_down/Reshape/shape/1:output:0:random_flip_1/random_flip_up_down/Reshape/shape/2:output:0:random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:21
/random_flip_1/random_flip_up_down/Reshape/shape
)random_flip_1/random_flip_up_down/ReshapeReshape8random_flip_1/random_flip_up_down/random_uniform/Mul:z:08random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2+
)random_flip_1/random_flip_up_down/ReshapeЩ
'random_flip_1/random_flip_up_down/RoundRound2random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2)
'random_flip_1/random_flip_up_down/RoundЎ
0random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:22
0random_flip_1/random_flip_up_down/ReverseV2/axis
+random_flip_1/random_flip_up_down/ReverseV2	ReverseV2=random_flip_1/random_flip_up_down/control_dependency:output:09random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2-
+random_flip_1/random_flip_up_down/ReverseV2є
%random_flip_1/random_flip_up_down/mulMul+random_flip_1/random_flip_up_down/Round:y:04random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/mul
'random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_flip_1/random_flip_up_down/sub/xю
%random_flip_1/random_flip_up_down/subSub0random_flip_1/random_flip_up_down/sub/x:output:0+random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/subџ
'random_flip_1/random_flip_up_down/mul_1Mul)random_flip_1/random_flip_up_down/sub:z:0=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2)
'random_flip_1/random_flip_up_down/mul_1ы
%random_flip_1/random_flip_up_down/addAddV2)random_flip_1/random_flip_up_down/mul:z:0+random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/add
random_rotation_1/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_1/Shape
%random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_rotation_1/strided_slice/stack
'random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_1
'random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_2Ю
random_rotation_1/strided_sliceStridedSlice random_rotation_1/Shape:output:0.random_rotation_1/strided_slice/stack:output:00random_rotation_1/strided_slice/stack_1:output:00random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation_1/strided_slice
'random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_1/stack 
)random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_1 
)random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_2и
!random_rotation_1/strided_slice_1StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_1/stack:output:02random_rotation_1/strided_slice_1/stack_1:output:02random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_1
random_rotation_1/CastCast*random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast
'random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_2/stack 
)random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_1 
)random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_2и
!random_rotation_1/strided_slice_2StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_2/stack:output:02random_rotation_1/strided_slice_2/stack_1:output:02random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_2
random_rotation_1/Cast_1Cast*random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast_1Д
(random_rotation_1/stateful_uniform/shapePack(random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:2*
(random_rotation_1/stateful_uniform/shape
&random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|й П2(
&random_rotation_1/stateful_uniform/min
&random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|й ?2(
&random_rotation_1/stateful_uniform/maxО
<random_rotation_1/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<random_rotation_1/stateful_uniform/StatefulUniform/algorithmъ
2random_rotation_1/stateful_uniform/StatefulUniformStatefulUniform;random_rotation_1_stateful_uniform_statefuluniform_resourceErandom_rotation_1/stateful_uniform/StatefulUniform/algorithm:output:01random_rotation_1/stateful_uniform/shape:output:0*#
_output_shapes
:џџџџџџџџџ*
shape_dtype024
2random_rotation_1/stateful_uniform/StatefulUniformк
&random_rotation_1/stateful_uniform/subSub/random_rotation_1/stateful_uniform/max:output:0/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2(
&random_rotation_1/stateful_uniform/subю
&random_rotation_1/stateful_uniform/mulMul;random_rotation_1/stateful_uniform/StatefulUniform:output:0*random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_1/stateful_uniform/mulк
"random_rotation_1/stateful_uniformAdd*random_rotation_1/stateful_uniform/mul:z:0/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2$
"random_rotation_1/stateful_uniform
'random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation_1/rotation_matrix/sub/yЦ
%random_rotation_1/rotation_matrix/subSubrandom_rotation_1/Cast_1:y:00random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation_1/rotation_matrix/subЋ
%random_rotation_1/rotation_matrix/CosCos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/Cos
)random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_1/yЬ
'random_rotation_1/rotation_matrix/sub_1Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_1л
%random_rotation_1/rotation_matrix/mulMul)random_rotation_1/rotation_matrix/Cos:y:0+random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/mulЋ
%random_rotation_1/rotation_matrix/SinSin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/Sin
)random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_2/yЪ
'random_rotation_1/rotation_matrix/sub_2Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_2п
'random_rotation_1/rotation_matrix/mul_1Mul)random_rotation_1/rotation_matrix/Sin:y:0+random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_1п
'random_rotation_1/rotation_matrix/sub_3Sub)random_rotation_1/rotation_matrix/mul:z:0+random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_3п
'random_rotation_1/rotation_matrix/sub_4Sub)random_rotation_1/rotation_matrix/sub:z:0+random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_4
+random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation_1/rotation_matrix/truediv/yђ
)random_rotation_1/rotation_matrix/truedivRealDiv+random_rotation_1/rotation_matrix/sub_4:z:04random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2+
)random_rotation_1/rotation_matrix/truediv
)random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_5/yЪ
'random_rotation_1/rotation_matrix/sub_5Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_5Џ
'random_rotation_1/rotation_matrix/Sin_1Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_1
)random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_6/yЬ
'random_rotation_1/rotation_matrix/sub_6Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_6с
'random_rotation_1/rotation_matrix/mul_2Mul+random_rotation_1/rotation_matrix/Sin_1:y:0+random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_2Џ
'random_rotation_1/rotation_matrix/Cos_1Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_1
)random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_7/yЪ
'random_rotation_1/rotation_matrix/sub_7Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_7с
'random_rotation_1/rotation_matrix/mul_3Mul+random_rotation_1/rotation_matrix/Cos_1:y:0+random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_3п
%random_rotation_1/rotation_matrix/addAddV2+random_rotation_1/rotation_matrix/mul_2:z:0+random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/addп
'random_rotation_1/rotation_matrix/sub_8Sub+random_rotation_1/rotation_matrix/sub_5:z:0)random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_8Ѓ
-random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-random_rotation_1/rotation_matrix/truediv_1/yј
+random_rotation_1/rotation_matrix/truediv_1RealDiv+random_rotation_1/rotation_matrix/sub_8:z:06random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2-
+random_rotation_1/rotation_matrix/truediv_1Ј
'random_rotation_1/rotation_matrix/ShapeShape&random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:2)
'random_rotation_1/rotation_matrix/ShapeИ
5random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_rotation_1/rotation_matrix/strided_slice/stackМ
7random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_1М
7random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_2Ў
/random_rotation_1/rotation_matrix/strided_sliceStridedSlice0random_rotation_1/rotation_matrix/Shape:output:0>random_rotation_1/rotation_matrix/strided_slice/stack:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_rotation_1/rotation_matrix/strided_sliceЏ
'random_rotation_1/rotation_matrix/Cos_2Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_2У
7random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_1/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_1StridedSlice+random_rotation_1/rotation_matrix/Cos_2:y:0@random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_1Џ
'random_rotation_1/rotation_matrix/Sin_2Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_2У
7random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_2/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_2StridedSlice+random_rotation_1/rotation_matrix/Sin_2:y:0@random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_2У
%random_rotation_1/rotation_matrix/NegNeg:random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/NegУ
7random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_3/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2х
1random_rotation_1/rotation_matrix/strided_slice_3StridedSlice-random_rotation_1/rotation_matrix/truediv:z:0@random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_3Џ
'random_rotation_1/rotation_matrix/Sin_3Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_3У
7random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_4/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_4StridedSlice+random_rotation_1/rotation_matrix/Sin_3:y:0@random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_4Џ
'random_rotation_1/rotation_matrix/Cos_3Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_3У
7random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_5/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_5StridedSlice+random_rotation_1/rotation_matrix/Cos_3:y:0@random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_5У
7random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_6/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2ч
1random_rotation_1/rotation_matrix/strided_slice_6StridedSlice/random_rotation_1/rotation_matrix/truediv_1:z:0@random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_6 
-random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/zeros/mul/yє
+random_rotation_1/rotation_matrix/zeros/mulMul8random_rotation_1/rotation_matrix/strided_slice:output:06random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2-
+random_rotation_1/rotation_matrix/zeros/mulЃ
.random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш20
.random_rotation_1/rotation_matrix/zeros/Less/yя
,random_rotation_1/rotation_matrix/zeros/LessLess/random_rotation_1/rotation_matrix/zeros/mul:z:07random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_1/rotation_matrix/zeros/LessІ
0random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
0random_rotation_1/rotation_matrix/zeros/packed/1
.random_rotation_1/rotation_matrix/zeros/packedPack8random_rotation_1/rotation_matrix/strided_slice:output:09random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.random_rotation_1/rotation_matrix/zeros/packedЃ
-random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-random_rotation_1/rotation_matrix/zeros/Const§
'random_rotation_1/rotation_matrix/zerosFill7random_rotation_1/rotation_matrix/zeros/packed:output:06random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/zeros 
-random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/concat/axisм
(random_rotation_1/rotation_matrix/concatConcatV2:random_rotation_1/rotation_matrix/strided_slice_1:output:0)random_rotation_1/rotation_matrix/Neg:y:0:random_rotation_1/rotation_matrix/strided_slice_3:output:0:random_rotation_1/rotation_matrix/strided_slice_4:output:0:random_rotation_1/rotation_matrix/strided_slice_5:output:0:random_rotation_1/rotation_matrix/strided_slice_6:output:00random_rotation_1/rotation_matrix/zeros:output:06random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2*
(random_rotation_1/rotation_matrix/concat
!random_rotation_1/transform/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2#
!random_rotation_1/transform/ShapeЌ
/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation_1/transform/strided_slice/stackА
1random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_1А
1random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_2і
)random_rotation_1/transform/strided_sliceStridedSlice*random_rotation_1/transform/Shape:output:08random_rotation_1/transform/strided_slice/stack:output:0:random_rotation_1/transform/strided_slice/stack_1:output:0:random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)random_rotation_1/transform/strided_slice
&random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_rotation_1/transform/fill_valueЦ
6random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3)random_flip_1/random_flip_up_down/add:z:01random_rotation_1/rotation_matrix/concat:output:02random_rotation_1/transform/strided_slice:output:0/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR28
6random_rotation_1/transform/ImageProjectiveTransformV3Ѕ
random_zoom_1/ShapeShapeKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_1/Shape
!random_zoom_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!random_zoom_1/strided_slice/stack
#random_zoom_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice/stack_1
#random_zoom_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice/stack_2Ж
random_zoom_1/strided_sliceStridedSlicerandom_zoom_1/Shape:output:0*random_zoom_1/strided_slice/stack:output:0,random_zoom_1/strided_slice/stack_1:output:0,random_zoom_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice
#random_zoom_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice_1/stack
%random_zoom_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_1/stack_1
%random_zoom_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_1/stack_2Р
random_zoom_1/strided_slice_1StridedSlicerandom_zoom_1/Shape:output:0,random_zoom_1/strided_slice_1/stack:output:0.random_zoom_1/strided_slice_1/stack_1:output:0.random_zoom_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice_1
random_zoom_1/CastCast&random_zoom_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_1/Cast
#random_zoom_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice_2/stack
%random_zoom_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_2/stack_1
%random_zoom_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_2/stack_2Р
random_zoom_1/strided_slice_2StridedSlicerandom_zoom_1/Shape:output:0,random_zoom_1/strided_slice_2/stack:output:0.random_zoom_1/strided_slice_2/stack_1:output:0.random_zoom_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice_2
random_zoom_1/Cast_1Cast&random_zoom_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_1/Cast_1
&random_zoom_1/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_1/stateful_uniform/shape/1й
$random_zoom_1/stateful_uniform/shapePack$random_zoom_1/strided_slice:output:0/random_zoom_1/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$random_zoom_1/stateful_uniform/shape
"random_zoom_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2$
"random_zoom_1/stateful_uniform/min
"random_zoom_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2$
"random_zoom_1/stateful_uniform/maxЖ
8random_zoom_1/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2:
8random_zoom_1/stateful_uniform/StatefulUniform/algorithmк
.random_zoom_1/stateful_uniform/StatefulUniformStatefulUniform7random_zoom_1_stateful_uniform_statefuluniform_resourceArandom_zoom_1/stateful_uniform/StatefulUniform/algorithm:output:0-random_zoom_1/stateful_uniform/shape:output:0*'
_output_shapes
:џџџџџџџџџ*
shape_dtype020
.random_zoom_1/stateful_uniform/StatefulUniformЪ
"random_zoom_1/stateful_uniform/subSub+random_zoom_1/stateful_uniform/max:output:0+random_zoom_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_zoom_1/stateful_uniform/subт
"random_zoom_1/stateful_uniform/mulMul7random_zoom_1/stateful_uniform/StatefulUniform:output:0&random_zoom_1/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"random_zoom_1/stateful_uniform/mulЮ
random_zoom_1/stateful_uniformAdd&random_zoom_1/stateful_uniform/mul:z:0+random_zoom_1/stateful_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
random_zoom_1/stateful_uniformx
random_zoom_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
random_zoom_1/concat/axisп
random_zoom_1/concatConcatV2"random_zoom_1/stateful_uniform:z:0"random_zoom_1/stateful_uniform:z:0"random_zoom_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
random_zoom_1/concat
random_zoom_1/zoom_matrix/ShapeShaperandom_zoom_1/concat:output:0*
T0*
_output_shapes
:2!
random_zoom_1/zoom_matrix/ShapeЈ
-random_zoom_1/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_zoom_1/zoom_matrix/strided_slice/stackЌ
/random_zoom_1/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_1/zoom_matrix/strided_slice/stack_1Ќ
/random_zoom_1/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_1/zoom_matrix/strided_slice/stack_2ў
'random_zoom_1/zoom_matrix/strided_sliceStridedSlice(random_zoom_1/zoom_matrix/Shape:output:06random_zoom_1/zoom_matrix/strided_slice/stack:output:08random_zoom_1/zoom_matrix/strided_slice/stack_1:output:08random_zoom_1/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'random_zoom_1/zoom_matrix/strided_slice
random_zoom_1/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
random_zoom_1/zoom_matrix/sub/yЊ
random_zoom_1/zoom_matrix/subSubrandom_zoom_1/Cast_1:y:0(random_zoom_1/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
random_zoom_1/zoom_matrix/sub
#random_zoom_1/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#random_zoom_1/zoom_matrix/truediv/yУ
!random_zoom_1/zoom_matrix/truedivRealDiv!random_zoom_1/zoom_matrix/sub:z:0,random_zoom_1/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2#
!random_zoom_1/zoom_matrix/truedivЗ
/random_zoom_1/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_1/zoom_matrix/strided_slice_1/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_1/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_1/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_1/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_1/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_1
!random_zoom_1/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_1/xл
random_zoom_1/zoom_matrix/sub_1Sub*random_zoom_1/zoom_matrix/sub_1/x:output:02random_zoom_1/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/sub_1У
random_zoom_1/zoom_matrix/mulMul%random_zoom_1/zoom_matrix/truediv:z:0#random_zoom_1/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
random_zoom_1/zoom_matrix/mul
!random_zoom_1/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_2/yЎ
random_zoom_1/zoom_matrix/sub_2Subrandom_zoom_1/Cast:y:0*random_zoom_1/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2!
random_zoom_1/zoom_matrix/sub_2
%random_zoom_1/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%random_zoom_1/zoom_matrix/truediv_1/yЫ
#random_zoom_1/zoom_matrix/truediv_1RealDiv#random_zoom_1/zoom_matrix/sub_2:z:0.random_zoom_1/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom_1/zoom_matrix/truediv_1З
/random_zoom_1/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_1/zoom_matrix/strided_slice_2/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_2/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_2/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_2/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_2/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_2
!random_zoom_1/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_3/xл
random_zoom_1/zoom_matrix/sub_3Sub*random_zoom_1/zoom_matrix/sub_3/x:output:02random_zoom_1/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/sub_3Щ
random_zoom_1/zoom_matrix/mul_1Mul'random_zoom_1/zoom_matrix/truediv_1:z:0#random_zoom_1/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/mul_1З
/random_zoom_1/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_1/zoom_matrix/strided_slice_3/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_3/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_3/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_3/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_3/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_3
%random_zoom_1/zoom_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_1/zoom_matrix/zeros/mul/yд
#random_zoom_1/zoom_matrix/zeros/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:0.random_zoom_1/zoom_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom_1/zoom_matrix/zeros/mul
&random_zoom_1/zoom_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2(
&random_zoom_1/zoom_matrix/zeros/Less/yЯ
$random_zoom_1/zoom_matrix/zeros/LessLess'random_zoom_1/zoom_matrix/zeros/mul:z:0/random_zoom_1/zoom_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$random_zoom_1/zoom_matrix/zeros/Less
(random_zoom_1/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom_1/zoom_matrix/zeros/packed/1ы
&random_zoom_1/zoom_matrix/zeros/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:01random_zoom_1/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom_1/zoom_matrix/zeros/packed
%random_zoom_1/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%random_zoom_1/zoom_matrix/zeros/Constн
random_zoom_1/zoom_matrix/zerosFill/random_zoom_1/zoom_matrix/zeros/packed:output:0.random_zoom_1/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/zeros
'random_zoom_1/zoom_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_1/zoom_matrix/zeros_1/mul/yк
%random_zoom_1/zoom_matrix/zeros_1/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:00random_zoom_1/zoom_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2'
%random_zoom_1/zoom_matrix/zeros_1/mul
(random_zoom_1/zoom_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2*
(random_zoom_1/zoom_matrix/zeros_1/Less/yз
&random_zoom_1/zoom_matrix/zeros_1/LessLess)random_zoom_1/zoom_matrix/zeros_1/mul:z:01random_zoom_1/zoom_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2(
&random_zoom_1/zoom_matrix/zeros_1/Less
*random_zoom_1/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_1/zoom_matrix/zeros_1/packed/1ё
(random_zoom_1/zoom_matrix/zeros_1/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:03random_zoom_1/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_1/zoom_matrix/zeros_1/packed
'random_zoom_1/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_1/zoom_matrix/zeros_1/Constх
!random_zoom_1/zoom_matrix/zeros_1Fill1random_zoom_1/zoom_matrix/zeros_1/packed:output:00random_zoom_1/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!random_zoom_1/zoom_matrix/zeros_1З
/random_zoom_1/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_1/zoom_matrix/strided_slice_4/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_4/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_4/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_4/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_4/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_4
'random_zoom_1/zoom_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_1/zoom_matrix/zeros_2/mul/yк
%random_zoom_1/zoom_matrix/zeros_2/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:00random_zoom_1/zoom_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2'
%random_zoom_1/zoom_matrix/zeros_2/mul
(random_zoom_1/zoom_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2*
(random_zoom_1/zoom_matrix/zeros_2/Less/yз
&random_zoom_1/zoom_matrix/zeros_2/LessLess)random_zoom_1/zoom_matrix/zeros_2/mul:z:01random_zoom_1/zoom_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2(
&random_zoom_1/zoom_matrix/zeros_2/Less
*random_zoom_1/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_1/zoom_matrix/zeros_2/packed/1ё
(random_zoom_1/zoom_matrix/zeros_2/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:03random_zoom_1/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_1/zoom_matrix/zeros_2/packed
'random_zoom_1/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_1/zoom_matrix/zeros_2/Constх
!random_zoom_1/zoom_matrix/zeros_2Fill1random_zoom_1/zoom_matrix/zeros_2/packed:output:00random_zoom_1/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!random_zoom_1/zoom_matrix/zeros_2
%random_zoom_1/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_1/zoom_matrix/concat/axisэ
 random_zoom_1/zoom_matrix/concatConcatV22random_zoom_1/zoom_matrix/strided_slice_3:output:0(random_zoom_1/zoom_matrix/zeros:output:0!random_zoom_1/zoom_matrix/mul:z:0*random_zoom_1/zoom_matrix/zeros_1:output:02random_zoom_1/zoom_matrix/strided_slice_4:output:0#random_zoom_1/zoom_matrix/mul_1:z:0*random_zoom_1/zoom_matrix/zeros_2:output:0.random_zoom_1/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2"
 random_zoom_1/zoom_matrix/concatЙ
random_zoom_1/transform/ShapeShapeKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_1/transform/ShapeЄ
+random_zoom_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_zoom_1/transform/strided_slice/stackЈ
-random_zoom_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_1/transform/strided_slice/stack_1Ј
-random_zoom_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_1/transform/strided_slice/stack_2о
%random_zoom_1/transform/strided_sliceStridedSlice&random_zoom_1/transform/Shape:output:04random_zoom_1/transform/strided_slice/stack:output:06random_zoom_1/transform/strided_slice/stack_1:output:06random_zoom_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%random_zoom_1/transform/strided_slice
"random_zoom_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"random_zoom_1/transform/fill_valueа
2random_zoom_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Krandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0)random_zoom_1/zoom_matrix/concat:output:0.random_zoom_1/transform/strided_slice:output:0+random_zoom_1/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR24
2random_zoom_1/transform/ImageProjectiveTransformV3
IdentityIdentityGrandom_zoom_1/transform/ImageProjectiveTransformV3:transformed_images:03^random_rotation_1/stateful_uniform/StatefulUniform/^random_zoom_1/stateful_uniform/StatefulUniform*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::2h
2random_rotation_1/stateful_uniform/StatefulUniform2random_rotation_1/stateful_uniform/StatefulUniform2`
.random_zoom_1/stateful_uniform/StatefulUniform.random_zoom_1/stateful_uniform/StatefulUniform:f b
1
_output_shapes
:џџџџџџџџџ
-
_user_specified_namerandom_flip_1_input
Я
J
.__inference_sequential_1_layer_call_fn_1724597

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_17232012
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь	
о
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1723421

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
вS

G__inference_sequential_layer_call_and_return_conditional_losses_1723646
sequential_1_input
conv2d_1723598
conv2d_1723600
conv2d_1_1723605
conv2d_1_1723607
conv2d_2_1723612
conv2d_2_1723614
conv2d_3_1723619
conv2d_3_1723621
conv2d_4_1723626
conv2d_4_1723628
dense_1723635
dense_1723637
dense_1_1723640
dense_1_1723642
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂ conv2d_4/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall
sequential_1/PartitionedCallPartitionedCallsequential_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_17232012
sequential_1/PartitionedCalli
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/xЃ
rescaling/mulMul%sequential_1/PartitionedCall:output:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/add­
conv2d/StatefulPartitionedCallStatefulPartitionedCallrescaling/add:z:0conv2d_1723598conv2d_1723600*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_17233012 
conv2d/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_17233222
activation/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_17232102
max_pooling2d/PartitionedCallЪ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1723605conv2d_1_1723607*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_17233412"
 conv2d_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ}}@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_17233622
activation_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>>@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17232222!
max_pooling2d_1/PartitionedCallЬ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1723612conv2d_2_1723614*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_17233812"
 conv2d_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_17234022
activation_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17232342!
max_pooling2d_2/PartitionedCallЬ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1723619conv2d_3_1723621*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_17234212"
 conv2d_3/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_17234422
activation_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17232462!
max_pooling2d_3/PartitionedCallЬ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_1723626conv2d_4_1723628*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_17234612"
 conv2d_4/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_17234822
activation_4/PartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17232582!
max_pooling2d_4/PartitionedCall
dropout/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_17235082
dropout/PartitionedCallј
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17235272
flatten/PartitionedCallЎ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1723635dense_1723637*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17235462
dense/StatefulPartitionedCallН
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1723640dense_1_1723642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_17235732!
dense_1/StatefulPartitionedCallы
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџ::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesequential_1_input
љ	
н
D__inference_dense_1_layer_call_and_return_conditional_losses_1723573

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
`
D__inference_flatten_layer_call_and_return_conditional_losses_1723527

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ХЋ
Ь
I__inference_sequential_1_layer_call_and_return_conditional_losses_1723188

inputs?
;random_rotation_1_stateful_uniform_statefuluniform_resource;
7random_zoom_1_stateful_uniform_statefuluniform_resource
identityЂ2random_rotation_1/stateful_uniform/StatefulUniformЂ.random_zoom_1/stateful_uniform/StatefulUniformн
7random_flip_1/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:џџџџџџџџџ29
7random_flip_1/random_flip_left_right/control_dependencyШ
*random_flip_1/random_flip_left_right/ShapeShape@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2,
*random_flip_1/random_flip_left_right/ShapeО
8random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_1/random_flip_left_right/strided_slice/stackТ
:random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_1Т
:random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_2Р
2random_flip_1/random_flip_left_right/strided_sliceStridedSlice3random_flip_1/random_flip_left_right/Shape:output:0Arandom_flip_1/random_flip_left_right/strided_slice/stack:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2random_flip_1/random_flip_left_right/strided_sliceщ
9random_flip_1/random_flip_left_right/random_uniform/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2;
9random_flip_1/random_flip_left_right/random_uniform/shapeЗ
7random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7random_flip_1/random_flip_left_right/random_uniform/minЗ
7random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7random_flip_1/random_flip_left_right/random_uniform/max
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformBrandom_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02C
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformЕ
7random_flip_1/random_flip_left_right/random_uniform/MulMulJrandom_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0@random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ29
7random_flip_1/random_flip_left_right/random_uniform/MulЎ
4random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/1Ў
4random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/2Ў
4random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/3
2random_flip_1/random_flip_left_right/Reshape/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0=random_flip_1/random_flip_left_right/Reshape/shape/1:output:0=random_flip_1/random_flip_left_right/Reshape/shape/2:output:0=random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:24
2random_flip_1/random_flip_left_right/Reshape/shape
,random_flip_1/random_flip_left_right/ReshapeReshape;random_flip_1/random_flip_left_right/random_uniform/Mul:z:0;random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2.
,random_flip_1/random_flip_left_right/Reshapeв
*random_flip_1/random_flip_left_right/RoundRound5random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2,
*random_flip_1/random_flip_left_right/RoundД
3random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:25
3random_flip_1/random_flip_left_right/ReverseV2/axisЉ
.random_flip_1/random_flip_left_right/ReverseV2	ReverseV2@random_flip_1/random_flip_left_right/control_dependency:output:0<random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ20
.random_flip_1/random_flip_left_right/ReverseV2
(random_flip_1/random_flip_left_right/mulMul.random_flip_1/random_flip_left_right/Round:y:07random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/mul
*random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_flip_1/random_flip_left_right/sub/xњ
(random_flip_1/random_flip_left_right/subSub3random_flip_1/random_flip_left_right/sub/x:output:0.random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/sub
*random_flip_1/random_flip_left_right/mul_1Mul,random_flip_1/random_flip_left_right/sub:z:0@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2,
*random_flip_1/random_flip_left_right/mul_1ї
(random_flip_1/random_flip_left_right/addAddV2,random_flip_1/random_flip_left_right/mul:z:0.random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_1/random_flip_left_right/add
4random_flip_1/random_flip_up_down/control_dependencyIdentity,random_flip_1/random_flip_left_right/add:z:0*
T0*;
_class1
/-loc:@random_flip_1/random_flip_left_right/add*1
_output_shapes
:џџџџџџџџџ26
4random_flip_1/random_flip_up_down/control_dependencyП
'random_flip_1/random_flip_up_down/ShapeShape=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2)
'random_flip_1/random_flip_up_down/ShapeИ
5random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_flip_1/random_flip_up_down/strided_slice/stackМ
7random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_1М
7random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_2Ў
/random_flip_1/random_flip_up_down/strided_sliceStridedSlice0random_flip_1/random_flip_up_down/Shape:output:0>random_flip_1/random_flip_up_down/strided_slice/stack:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_flip_1/random_flip_up_down/strided_sliceр
6random_flip_1/random_flip_up_down/random_uniform/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6random_flip_1/random_flip_up_down/random_uniform/shapeБ
4random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4random_flip_1/random_flip_up_down/random_uniform/minБ
4random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4random_flip_1/random_flip_up_down/random_uniform/max
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniform?random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02@
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformЉ
4random_flip_1/random_flip_up_down/random_uniform/MulMulGrandom_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0=random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ26
4random_flip_1/random_flip_up_down/random_uniform/MulЈ
1random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/1Ј
1random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/2Ј
1random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/3
/random_flip_1/random_flip_up_down/Reshape/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0:random_flip_1/random_flip_up_down/Reshape/shape/1:output:0:random_flip_1/random_flip_up_down/Reshape/shape/2:output:0:random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:21
/random_flip_1/random_flip_up_down/Reshape/shape
)random_flip_1/random_flip_up_down/ReshapeReshape8random_flip_1/random_flip_up_down/random_uniform/Mul:z:08random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2+
)random_flip_1/random_flip_up_down/ReshapeЩ
'random_flip_1/random_flip_up_down/RoundRound2random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2)
'random_flip_1/random_flip_up_down/RoundЎ
0random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:22
0random_flip_1/random_flip_up_down/ReverseV2/axis
+random_flip_1/random_flip_up_down/ReverseV2	ReverseV2=random_flip_1/random_flip_up_down/control_dependency:output:09random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2-
+random_flip_1/random_flip_up_down/ReverseV2є
%random_flip_1/random_flip_up_down/mulMul+random_flip_1/random_flip_up_down/Round:y:04random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/mul
'random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_flip_1/random_flip_up_down/sub/xю
%random_flip_1/random_flip_up_down/subSub0random_flip_1/random_flip_up_down/sub/x:output:0+random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/subџ
'random_flip_1/random_flip_up_down/mul_1Mul)random_flip_1/random_flip_up_down/sub:z:0=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2)
'random_flip_1/random_flip_up_down/mul_1ы
%random_flip_1/random_flip_up_down/addAddV2)random_flip_1/random_flip_up_down/mul:z:0+random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2'
%random_flip_1/random_flip_up_down/add
random_rotation_1/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_1/Shape
%random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_rotation_1/strided_slice/stack
'random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_1
'random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_2Ю
random_rotation_1/strided_sliceStridedSlice random_rotation_1/Shape:output:0.random_rotation_1/strided_slice/stack:output:00random_rotation_1/strided_slice/stack_1:output:00random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation_1/strided_slice
'random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_1/stack 
)random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_1 
)random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_2и
!random_rotation_1/strided_slice_1StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_1/stack:output:02random_rotation_1/strided_slice_1/stack_1:output:02random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_1
random_rotation_1/CastCast*random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast
'random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_2/stack 
)random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_1 
)random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_2и
!random_rotation_1/strided_slice_2StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_2/stack:output:02random_rotation_1/strided_slice_2/stack_1:output:02random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_2
random_rotation_1/Cast_1Cast*random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast_1Д
(random_rotation_1/stateful_uniform/shapePack(random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:2*
(random_rotation_1/stateful_uniform/shape
&random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|й П2(
&random_rotation_1/stateful_uniform/min
&random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|й ?2(
&random_rotation_1/stateful_uniform/maxО
<random_rotation_1/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<random_rotation_1/stateful_uniform/StatefulUniform/algorithmъ
2random_rotation_1/stateful_uniform/StatefulUniformStatefulUniform;random_rotation_1_stateful_uniform_statefuluniform_resourceErandom_rotation_1/stateful_uniform/StatefulUniform/algorithm:output:01random_rotation_1/stateful_uniform/shape:output:0*#
_output_shapes
:џџџџџџџџџ*
shape_dtype024
2random_rotation_1/stateful_uniform/StatefulUniformк
&random_rotation_1/stateful_uniform/subSub/random_rotation_1/stateful_uniform/max:output:0/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2(
&random_rotation_1/stateful_uniform/subю
&random_rotation_1/stateful_uniform/mulMul;random_rotation_1/stateful_uniform/StatefulUniform:output:0*random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_1/stateful_uniform/mulк
"random_rotation_1/stateful_uniformAdd*random_rotation_1/stateful_uniform/mul:z:0/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2$
"random_rotation_1/stateful_uniform
'random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation_1/rotation_matrix/sub/yЦ
%random_rotation_1/rotation_matrix/subSubrandom_rotation_1/Cast_1:y:00random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation_1/rotation_matrix/subЋ
%random_rotation_1/rotation_matrix/CosCos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/Cos
)random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_1/yЬ
'random_rotation_1/rotation_matrix/sub_1Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_1л
%random_rotation_1/rotation_matrix/mulMul)random_rotation_1/rotation_matrix/Cos:y:0+random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/mulЋ
%random_rotation_1/rotation_matrix/SinSin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/Sin
)random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_2/yЪ
'random_rotation_1/rotation_matrix/sub_2Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_2п
'random_rotation_1/rotation_matrix/mul_1Mul)random_rotation_1/rotation_matrix/Sin:y:0+random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_1п
'random_rotation_1/rotation_matrix/sub_3Sub)random_rotation_1/rotation_matrix/mul:z:0+random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_3п
'random_rotation_1/rotation_matrix/sub_4Sub)random_rotation_1/rotation_matrix/sub:z:0+random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_4
+random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation_1/rotation_matrix/truediv/yђ
)random_rotation_1/rotation_matrix/truedivRealDiv+random_rotation_1/rotation_matrix/sub_4:z:04random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2+
)random_rotation_1/rotation_matrix/truediv
)random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_5/yЪ
'random_rotation_1/rotation_matrix/sub_5Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_5Џ
'random_rotation_1/rotation_matrix/Sin_1Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_1
)random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_6/yЬ
'random_rotation_1/rotation_matrix/sub_6Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_6с
'random_rotation_1/rotation_matrix/mul_2Mul+random_rotation_1/rotation_matrix/Sin_1:y:0+random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_2Џ
'random_rotation_1/rotation_matrix/Cos_1Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_1
)random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_1/rotation_matrix/sub_7/yЪ
'random_rotation_1/rotation_matrix/sub_7Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_7с
'random_rotation_1/rotation_matrix/mul_3Mul+random_rotation_1/rotation_matrix/Cos_1:y:0+random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/mul_3п
%random_rotation_1/rotation_matrix/addAddV2+random_rotation_1/rotation_matrix/mul_2:z:0+random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/addп
'random_rotation_1/rotation_matrix/sub_8Sub+random_rotation_1/rotation_matrix/sub_5:z:0)random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/sub_8Ѓ
-random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-random_rotation_1/rotation_matrix/truediv_1/yј
+random_rotation_1/rotation_matrix/truediv_1RealDiv+random_rotation_1/rotation_matrix/sub_8:z:06random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2-
+random_rotation_1/rotation_matrix/truediv_1Ј
'random_rotation_1/rotation_matrix/ShapeShape&random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:2)
'random_rotation_1/rotation_matrix/ShapeИ
5random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_rotation_1/rotation_matrix/strided_slice/stackМ
7random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_1М
7random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_2Ў
/random_rotation_1/rotation_matrix/strided_sliceStridedSlice0random_rotation_1/rotation_matrix/Shape:output:0>random_rotation_1/rotation_matrix/strided_slice/stack:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_rotation_1/rotation_matrix/strided_sliceЏ
'random_rotation_1/rotation_matrix/Cos_2Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_2У
7random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_1/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_1StridedSlice+random_rotation_1/rotation_matrix/Cos_2:y:0@random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_1Џ
'random_rotation_1/rotation_matrix/Sin_2Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_2У
7random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_2/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_2StridedSlice+random_rotation_1/rotation_matrix/Sin_2:y:0@random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_2У
%random_rotation_1/rotation_matrix/NegNeg:random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2'
%random_rotation_1/rotation_matrix/NegУ
7random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_3/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2х
1random_rotation_1/rotation_matrix/strided_slice_3StridedSlice-random_rotation_1/rotation_matrix/truediv:z:0@random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_3Џ
'random_rotation_1/rotation_matrix/Sin_3Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Sin_3У
7random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_4/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_4StridedSlice+random_rotation_1/rotation_matrix/Sin_3:y:0@random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_4Џ
'random_rotation_1/rotation_matrix/Cos_3Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/Cos_3У
7random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_5/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2у
1random_rotation_1/rotation_matrix/strided_slice_5StridedSlice+random_rotation_1/rotation_matrix/Cos_3:y:0@random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_5У
7random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_6/stackЧ
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Ч
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2ч
1random_rotation_1/rotation_matrix/strided_slice_6StridedSlice/random_rotation_1/rotation_matrix/truediv_1:z:0@random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_6 
-random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/zeros/mul/yє
+random_rotation_1/rotation_matrix/zeros/mulMul8random_rotation_1/rotation_matrix/strided_slice:output:06random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2-
+random_rotation_1/rotation_matrix/zeros/mulЃ
.random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш20
.random_rotation_1/rotation_matrix/zeros/Less/yя
,random_rotation_1/rotation_matrix/zeros/LessLess/random_rotation_1/rotation_matrix/zeros/mul:z:07random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_1/rotation_matrix/zeros/LessІ
0random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
0random_rotation_1/rotation_matrix/zeros/packed/1
.random_rotation_1/rotation_matrix/zeros/packedPack8random_rotation_1/rotation_matrix/strided_slice:output:09random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.random_rotation_1/rotation_matrix/zeros/packedЃ
-random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-random_rotation_1/rotation_matrix/zeros/Const§
'random_rotation_1/rotation_matrix/zerosFill7random_rotation_1/rotation_matrix/zeros/packed:output:06random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'random_rotation_1/rotation_matrix/zeros 
-random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/concat/axisм
(random_rotation_1/rotation_matrix/concatConcatV2:random_rotation_1/rotation_matrix/strided_slice_1:output:0)random_rotation_1/rotation_matrix/Neg:y:0:random_rotation_1/rotation_matrix/strided_slice_3:output:0:random_rotation_1/rotation_matrix/strided_slice_4:output:0:random_rotation_1/rotation_matrix/strided_slice_5:output:0:random_rotation_1/rotation_matrix/strided_slice_6:output:00random_rotation_1/rotation_matrix/zeros:output:06random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2*
(random_rotation_1/rotation_matrix/concat
!random_rotation_1/transform/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2#
!random_rotation_1/transform/ShapeЌ
/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation_1/transform/strided_slice/stackА
1random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_1А
1random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_2і
)random_rotation_1/transform/strided_sliceStridedSlice*random_rotation_1/transform/Shape:output:08random_rotation_1/transform/strided_slice/stack:output:0:random_rotation_1/transform/strided_slice/stack_1:output:0:random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)random_rotation_1/transform/strided_slice
&random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_rotation_1/transform/fill_valueЦ
6random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3)random_flip_1/random_flip_up_down/add:z:01random_rotation_1/rotation_matrix/concat:output:02random_rotation_1/transform/strided_slice:output:0/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR28
6random_rotation_1/transform/ImageProjectiveTransformV3Ѕ
random_zoom_1/ShapeShapeKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_1/Shape
!random_zoom_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!random_zoom_1/strided_slice/stack
#random_zoom_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice/stack_1
#random_zoom_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice/stack_2Ж
random_zoom_1/strided_sliceStridedSlicerandom_zoom_1/Shape:output:0*random_zoom_1/strided_slice/stack:output:0,random_zoom_1/strided_slice/stack_1:output:0,random_zoom_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice
#random_zoom_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice_1/stack
%random_zoom_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_1/stack_1
%random_zoom_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_1/stack_2Р
random_zoom_1/strided_slice_1StridedSlicerandom_zoom_1/Shape:output:0,random_zoom_1/strided_slice_1/stack:output:0.random_zoom_1/strided_slice_1/stack_1:output:0.random_zoom_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice_1
random_zoom_1/CastCast&random_zoom_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_1/Cast
#random_zoom_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_1/strided_slice_2/stack
%random_zoom_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_2/stack_1
%random_zoom_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_1/strided_slice_2/stack_2Р
random_zoom_1/strided_slice_2StridedSlicerandom_zoom_1/Shape:output:0,random_zoom_1/strided_slice_2/stack:output:0.random_zoom_1/strided_slice_2/stack_1:output:0.random_zoom_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_1/strided_slice_2
random_zoom_1/Cast_1Cast&random_zoom_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_1/Cast_1
&random_zoom_1/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_1/stateful_uniform/shape/1й
$random_zoom_1/stateful_uniform/shapePack$random_zoom_1/strided_slice:output:0/random_zoom_1/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$random_zoom_1/stateful_uniform/shape
"random_zoom_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL?2$
"random_zoom_1/stateful_uniform/min
"random_zoom_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2$
"random_zoom_1/stateful_uniform/maxЖ
8random_zoom_1/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2:
8random_zoom_1/stateful_uniform/StatefulUniform/algorithmк
.random_zoom_1/stateful_uniform/StatefulUniformStatefulUniform7random_zoom_1_stateful_uniform_statefuluniform_resourceArandom_zoom_1/stateful_uniform/StatefulUniform/algorithm:output:0-random_zoom_1/stateful_uniform/shape:output:0*'
_output_shapes
:џџџџџџџџџ*
shape_dtype020
.random_zoom_1/stateful_uniform/StatefulUniformЪ
"random_zoom_1/stateful_uniform/subSub+random_zoom_1/stateful_uniform/max:output:0+random_zoom_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_zoom_1/stateful_uniform/subт
"random_zoom_1/stateful_uniform/mulMul7random_zoom_1/stateful_uniform/StatefulUniform:output:0&random_zoom_1/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"random_zoom_1/stateful_uniform/mulЮ
random_zoom_1/stateful_uniformAdd&random_zoom_1/stateful_uniform/mul:z:0+random_zoom_1/stateful_uniform/min:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
random_zoom_1/stateful_uniformx
random_zoom_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
random_zoom_1/concat/axisп
random_zoom_1/concatConcatV2"random_zoom_1/stateful_uniform:z:0"random_zoom_1/stateful_uniform:z:0"random_zoom_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
random_zoom_1/concat
random_zoom_1/zoom_matrix/ShapeShaperandom_zoom_1/concat:output:0*
T0*
_output_shapes
:2!
random_zoom_1/zoom_matrix/ShapeЈ
-random_zoom_1/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_zoom_1/zoom_matrix/strided_slice/stackЌ
/random_zoom_1/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_1/zoom_matrix/strided_slice/stack_1Ќ
/random_zoom_1/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_1/zoom_matrix/strided_slice/stack_2ў
'random_zoom_1/zoom_matrix/strided_sliceStridedSlice(random_zoom_1/zoom_matrix/Shape:output:06random_zoom_1/zoom_matrix/strided_slice/stack:output:08random_zoom_1/zoom_matrix/strided_slice/stack_1:output:08random_zoom_1/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'random_zoom_1/zoom_matrix/strided_slice
random_zoom_1/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
random_zoom_1/zoom_matrix/sub/yЊ
random_zoom_1/zoom_matrix/subSubrandom_zoom_1/Cast_1:y:0(random_zoom_1/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
random_zoom_1/zoom_matrix/sub
#random_zoom_1/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#random_zoom_1/zoom_matrix/truediv/yУ
!random_zoom_1/zoom_matrix/truedivRealDiv!random_zoom_1/zoom_matrix/sub:z:0,random_zoom_1/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2#
!random_zoom_1/zoom_matrix/truedivЗ
/random_zoom_1/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_1/zoom_matrix/strided_slice_1/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_1/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_1/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_1/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_1/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_1
!random_zoom_1/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_1/xл
random_zoom_1/zoom_matrix/sub_1Sub*random_zoom_1/zoom_matrix/sub_1/x:output:02random_zoom_1/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/sub_1У
random_zoom_1/zoom_matrix/mulMul%random_zoom_1/zoom_matrix/truediv:z:0#random_zoom_1/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
random_zoom_1/zoom_matrix/mul
!random_zoom_1/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_2/yЎ
random_zoom_1/zoom_matrix/sub_2Subrandom_zoom_1/Cast:y:0*random_zoom_1/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2!
random_zoom_1/zoom_matrix/sub_2
%random_zoom_1/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%random_zoom_1/zoom_matrix/truediv_1/yЫ
#random_zoom_1/zoom_matrix/truediv_1RealDiv#random_zoom_1/zoom_matrix/sub_2:z:0.random_zoom_1/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom_1/zoom_matrix/truediv_1З
/random_zoom_1/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_1/zoom_matrix/strided_slice_2/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_2/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_2/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_2/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_2/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_2
!random_zoom_1/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_1/zoom_matrix/sub_3/xл
random_zoom_1/zoom_matrix/sub_3Sub*random_zoom_1/zoom_matrix/sub_3/x:output:02random_zoom_1/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/sub_3Щ
random_zoom_1/zoom_matrix/mul_1Mul'random_zoom_1/zoom_matrix/truediv_1:z:0#random_zoom_1/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/mul_1З
/random_zoom_1/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_1/zoom_matrix/strided_slice_3/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_3/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_3/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_3/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_3/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_3
%random_zoom_1/zoom_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_1/zoom_matrix/zeros/mul/yд
#random_zoom_1/zoom_matrix/zeros/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:0.random_zoom_1/zoom_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom_1/zoom_matrix/zeros/mul
&random_zoom_1/zoom_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2(
&random_zoom_1/zoom_matrix/zeros/Less/yЯ
$random_zoom_1/zoom_matrix/zeros/LessLess'random_zoom_1/zoom_matrix/zeros/mul:z:0/random_zoom_1/zoom_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$random_zoom_1/zoom_matrix/zeros/Less
(random_zoom_1/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom_1/zoom_matrix/zeros/packed/1ы
&random_zoom_1/zoom_matrix/zeros/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:01random_zoom_1/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom_1/zoom_matrix/zeros/packed
%random_zoom_1/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%random_zoom_1/zoom_matrix/zeros/Constн
random_zoom_1/zoom_matrix/zerosFill/random_zoom_1/zoom_matrix/zeros/packed:output:0.random_zoom_1/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
random_zoom_1/zoom_matrix/zeros
'random_zoom_1/zoom_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_1/zoom_matrix/zeros_1/mul/yк
%random_zoom_1/zoom_matrix/zeros_1/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:00random_zoom_1/zoom_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2'
%random_zoom_1/zoom_matrix/zeros_1/mul
(random_zoom_1/zoom_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2*
(random_zoom_1/zoom_matrix/zeros_1/Less/yз
&random_zoom_1/zoom_matrix/zeros_1/LessLess)random_zoom_1/zoom_matrix/zeros_1/mul:z:01random_zoom_1/zoom_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2(
&random_zoom_1/zoom_matrix/zeros_1/Less
*random_zoom_1/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_1/zoom_matrix/zeros_1/packed/1ё
(random_zoom_1/zoom_matrix/zeros_1/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:03random_zoom_1/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_1/zoom_matrix/zeros_1/packed
'random_zoom_1/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_1/zoom_matrix/zeros_1/Constх
!random_zoom_1/zoom_matrix/zeros_1Fill1random_zoom_1/zoom_matrix/zeros_1/packed:output:00random_zoom_1/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!random_zoom_1/zoom_matrix/zeros_1З
/random_zoom_1/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_1/zoom_matrix/strided_slice_4/stackЛ
1random_zoom_1/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_1/zoom_matrix/strided_slice_4/stack_1Л
1random_zoom_1/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_1/zoom_matrix/strided_slice_4/stack_2Х
)random_zoom_1/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_1/concat:output:08random_zoom_1/zoom_matrix/strided_slice_4/stack:output:0:random_zoom_1/zoom_matrix/strided_slice_4/stack_1:output:0:random_zoom_1/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_1/zoom_matrix/strided_slice_4
'random_zoom_1/zoom_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_1/zoom_matrix/zeros_2/mul/yк
%random_zoom_1/zoom_matrix/zeros_2/mulMul0random_zoom_1/zoom_matrix/strided_slice:output:00random_zoom_1/zoom_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2'
%random_zoom_1/zoom_matrix/zeros_2/mul
(random_zoom_1/zoom_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2*
(random_zoom_1/zoom_matrix/zeros_2/Less/yз
&random_zoom_1/zoom_matrix/zeros_2/LessLess)random_zoom_1/zoom_matrix/zeros_2/mul:z:01random_zoom_1/zoom_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2(
&random_zoom_1/zoom_matrix/zeros_2/Less
*random_zoom_1/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_1/zoom_matrix/zeros_2/packed/1ё
(random_zoom_1/zoom_matrix/zeros_2/packedPack0random_zoom_1/zoom_matrix/strided_slice:output:03random_zoom_1/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_1/zoom_matrix/zeros_2/packed
'random_zoom_1/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_1/zoom_matrix/zeros_2/Constх
!random_zoom_1/zoom_matrix/zeros_2Fill1random_zoom_1/zoom_matrix/zeros_2/packed:output:00random_zoom_1/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!random_zoom_1/zoom_matrix/zeros_2
%random_zoom_1/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_1/zoom_matrix/concat/axisэ
 random_zoom_1/zoom_matrix/concatConcatV22random_zoom_1/zoom_matrix/strided_slice_3:output:0(random_zoom_1/zoom_matrix/zeros:output:0!random_zoom_1/zoom_matrix/mul:z:0*random_zoom_1/zoom_matrix/zeros_1:output:02random_zoom_1/zoom_matrix/strided_slice_4:output:0#random_zoom_1/zoom_matrix/mul_1:z:0*random_zoom_1/zoom_matrix/zeros_2:output:0.random_zoom_1/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2"
 random_zoom_1/zoom_matrix/concatЙ
random_zoom_1/transform/ShapeShapeKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_1/transform/ShapeЄ
+random_zoom_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_zoom_1/transform/strided_slice/stackЈ
-random_zoom_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_1/transform/strided_slice/stack_1Ј
-random_zoom_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_1/transform/strided_slice/stack_2о
%random_zoom_1/transform/strided_sliceStridedSlice&random_zoom_1/transform/Shape:output:04random_zoom_1/transform/strided_slice/stack:output:06random_zoom_1/transform/strided_slice/stack_1:output:06random_zoom_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%random_zoom_1/transform/strided_slice
"random_zoom_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"random_zoom_1/transform/fill_valueа
2random_zoom_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Krandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0)random_zoom_1/zoom_matrix/concat:output:0.random_zoom_1/transform/strided_slice:output:0+random_zoom_1/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR24
2random_zoom_1/transform/ImageProjectiveTransformV3
IdentityIdentityGrandom_zoom_1/transform/ImageProjectiveTransformV3:transformed_images:03^random_rotation_1/stateful_uniform/StatefulUniform/^random_zoom_1/stateful_uniform/StatefulUniform*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::2h
2random_rotation_1/stateful_uniform/StatefulUniform2random_rotation_1/stateful_uniform/StatefulUniform2`
.random_zoom_1/stateful_uniform/StatefulUniform.random_zoom_1/stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1723246

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

}
(__inference_conv2d_layer_call_fn_1724616

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_17233012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџўў@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
W
.__inference_sequential_1_layer_call_fn_1723204
random_flip_1_input
identityщ
PartitionedCallPartitionedCallrandom_flip_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_17232012
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџ:f b
1
_output_shapes
:џџџџџџџџџ
-
_user_specified_namerandom_flip_1_input
ч
|
'__inference_dense_layer_call_fn_1724800

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17235462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1723210

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѕ	
л
B__inference_dense_layer_call_and_return_conditional_losses_1723546

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з
e
I__inference_activation_4_layer_call_and_return_conditional_losses_1724737

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ъ

и
,__inference_sequential_layer_call_fn_1724296

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityЂStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*0
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_17237092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
гS
 	
G__inference_sequential_layer_call_and_return_conditional_losses_1724259

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂconv2d_4/BiasAdd/ReadVariableOpЂconv2d_4/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpi
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/x
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
rescaling/addЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpЦ
conv2d/Conv2DConv2Drescaling/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@*
paddingVALID*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpІ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2
conv2d/BiasAdd
activation/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2
activation/ReluХ
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@*
paddingVALID*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2
conv2d_1/BiasAdd
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ}}@2
activation_1/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpй
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_2/BiasAdd
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
activation_2/ReluЫ
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolА
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpй
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_3/Conv2DЇ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpЌ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_3/BiasAdd
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
activation_3/ReluЫ
max_pooling2d_3/MaxPoolMaxPoolactivation_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolА
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOpй
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_4/Conv2DЇ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpЌ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_4/BiasAdd
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
activation_4/ReluЫ
max_pooling2d_4/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool
dropout/IdentityIdentity max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 	  2
flatten/Const
flatten/ReshapeReshapedropout/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten/ReshapeЁ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

dense/ReluІ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_1/SoftmaxЖ
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџ::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж
K
/__inference_max_pooling2d_layer_call_fn_1723216

inputs
identityі
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_17232102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
н
c
G__inference_activation_layer_call_and_return_conditional_losses_1724621

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџўў@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџўў@:Y U
1
_output_shapes
:џџџџџџџџџўў@
 
_user_specified_nameinputs
ч
b
D__inference_dropout_layer_call_and_return_conditional_losses_1724759

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
К
M
1__inference_max_pooling2d_1_layer_call_fn_1723228

inputs
identityј
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17232222
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ	
н
D__inference_dense_1_layer_call_and_return_conditional_losses_1724811

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


Н
%__inference_signature_wrapper_1723876
sequential_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallsequential_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*0
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *+
f&R$
"__inference__wrapped_model_17226802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesequential_1_input
К
M
1__inference_max_pooling2d_2_layer_call_fn_1723240

inputs
identityј
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17232342
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ќ

Ф
,__inference_sequential_layer_call_fn_1723833
sequential_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallsequential_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*0
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

 рЋE8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_17238022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesequential_1_input
Н

.__inference_sequential_1_layer_call_fn_1723195
random_flip_1_input
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

 рЋE8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_17231882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:џџџџџџџџџ
-
_user_specified_namerandom_flip_1_input
Т
c
D__inference_dropout_layer_call_and_return_conditional_losses_1724754

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
є	
м
C__inference_conv2d_layer_call_and_return_conditional_losses_1724607

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџўў@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1723258

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь	
о
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1723461

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ъ
serving_defaultЖ
[
sequential_1_inputE
$serving_default_sequential_1_input:0џџџџџџџџџ;
dense_10
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:и

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-5
layer-19
layer_with_weights-6
layer-20
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"ф
_tf_keras_sequentialФ{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_1_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_1_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomZoom", "config": {"name": "random_zoom_1", "trainable": true, "dtype": "float32", "height_factor": 0.2, "width_factor": null, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}]}}, {"class_name": "Rescaling", "config": {"name": "rescaling", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_1_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_1_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomZoom", "config": {"name": "random_zoom_1", "trainable": true, "dtype": "float32", "height_factor": 0.2, "width_factor": null, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}]}}, {"class_name": "Rescaling", "config": {"name": "rescaling", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Х
layer-0
layer-1
layer-2
trainable_variables
 regularization_losses
!	variables
"	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_sequentialю{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_1_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomZoom", "config": {"name": "random_zoom_1", "trainable": true, "dtype": "float32", "height_factor": 0.2, "width_factor": null, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_1_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomZoom", "config": {"name": "random_zoom_1", "trainable": true, "dtype": "float32", "height_factor": 0.2, "width_factor": null, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}]}}}
ь
#	keras_api"к
_tf_keras_layerР{"class_name": "Rescaling", "name": "rescaling", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "rescaling", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}
ј


$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses"б	
_tf_keras_layerЗ	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}
г
*trainable_variables
+regularization_losses
,	variables
-	keras_api
Ѓ__call__
+Є&call_and_return_all_conditional_losses"Т
_tf_keras_layerЈ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
§
.trainable_variables
/regularization_losses
0	variables
1	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses"ь
_tf_keras_layerв{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
љ	

2kernel
3bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 127, 127, 64]}}
з
8trainable_variables
9regularization_losses
:	variables
;	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses"Ц
_tf_keras_layerЌ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}

<trainable_variables
=regularization_losses
>	variables
?	keras_api
Ћ__call__
+Ќ&call_and_return_all_conditional_losses"№
_tf_keras_layerж{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ї	

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62, 62, 64]}}
з
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
Џ__call__
+А&call_and_return_all_conditional_losses"Ц
_tf_keras_layerЌ{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}

Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"№
_tf_keras_layerж{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ї	

Nkernel
Obias
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 64]}}
з
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"Ц
_tf_keras_layerЌ{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}

Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
З__call__
+И&call_and_return_all_conditional_losses"№
_tf_keras_layerж{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ї	

\kernel
]bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
з
btrainable_variables
cregularization_losses
d	variables
e	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"Ц
_tf_keras_layerЌ{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}

ftrainable_variables
gregularization_losses
h	variables
i	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"№
_tf_keras_layerж{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
у
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
ф
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ѓ

rkernel
sbias
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"Ь
_tf_keras_layerВ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
ї

xkernel
ybias
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ю
~iter

beta_1
beta_2

decay
learning_rate$m%m2m3m@mAmNmOm\m]mrmsmxmym$v%v2v3v@vAvNvOv\v]vrvsvxvyv"
	optimizer

$0
%1
22
33
@4
A5
N6
O7
\8
]9
r10
s11
x12
y13"
trackable_list_wrapper
 "
trackable_list_wrapper

$0
%1
22
33
@4
A5
N6
O7
\8
]9
r10
s11
x12
y13"
trackable_list_wrapper
г
metrics
 layer_regularization_losses
layer_metrics
trainable_variables
layers
regularization_losses
	variables
non_trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Чserving_default"
signature_map

	_rng
	keras_api"і
_tf_keras_layerм{"class_name": "RandomFlip", "name": "random_flip_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_flip_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
П
	_rng
	keras_api"Ё
_tf_keras_layer{"class_name": "RandomRotation", "name": "random_rotation_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}
а
	_rng
	keras_api"В
_tf_keras_layer{"class_name": "RandomZoom", "name": "random_zoom_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_zoom_1", "trainable": true, "dtype": "float32", "height_factor": 0.2, "width_factor": null, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
metrics
 layer_regularization_losses
layer_metrics
trainable_variables
layers
 regularization_losses
!	variables
non_trainable_variables
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
':%@2conv2d/kernel
:@2conv2d/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
Е
metrics
 layer_regularization_losses
layer_metrics
&trainable_variables
layers
'regularization_losses
(	variables
non_trainable_variables
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
metrics
 layer_regularization_losses
layer_metrics
*trainable_variables
layers
+regularization_losses
,	variables
non_trainable_variables
Ѓ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
metrics
 layer_regularization_losses
layer_metrics
.trainable_variables
 layers
/regularization_losses
0	variables
Ёnon_trainable_variables
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
Е
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
4trainable_variables
Ѕlayers
5regularization_losses
6	variables
Іnon_trainable_variables
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
8trainable_variables
Њlayers
9regularization_losses
:	variables
Ћnon_trainable_variables
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
<trainable_variables
Џlayers
=regularization_losses
>	variables
Аnon_trainable_variables
Ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
Е
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
Btrainable_variables
Дlayers
Cregularization_losses
D	variables
Еnon_trainable_variables
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
Ftrainable_variables
Йlayers
Gregularization_losses
H	variables
Кnon_trainable_variables
Џ__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
Jtrainable_variables
Оlayers
Kregularization_losses
L	variables
Пnon_trainable_variables
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
Е
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Ptrainable_variables
Уlayers
Qregularization_losses
R	variables
Фnon_trainable_variables
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
Ttrainable_variables
Шlayers
Uregularization_losses
V	variables
Щnon_trainable_variables
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
Xtrainable_variables
Эlayers
Yregularization_losses
Z	variables
Юnon_trainable_variables
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
Е
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
^trainable_variables
вlayers
_regularization_losses
`	variables
гnon_trainable_variables
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
дmetrics
 еlayer_regularization_losses
жlayer_metrics
btrainable_variables
зlayers
cregularization_losses
d	variables
иnon_trainable_variables
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
йmetrics
 кlayer_regularization_losses
лlayer_metrics
ftrainable_variables
мlayers
gregularization_losses
h	variables
нnon_trainable_variables
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
оmetrics
 пlayer_regularization_losses
рlayer_metrics
jtrainable_variables
сlayers
kregularization_losses
l	variables
тnon_trainable_variables
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
уmetrics
 фlayer_regularization_losses
хlayer_metrics
ntrainable_variables
цlayers
oregularization_losses
p	variables
чnon_trainable_variables
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 :
2dense/kernel
:2
dense/bias
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
Е
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
ttrainable_variables
ыlayers
uregularization_losses
v	variables
ьnon_trainable_variables
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
!:	
2dense_1/kernel
:
2dense_1/bias
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
Е
эmetrics
 юlayer_regularization_losses
яlayer_metrics
ztrainable_variables
№layers
{regularization_losses
|	variables
ёnon_trainable_variables
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
ђ0
ѓ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
О
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
 "
trackable_list_wrapper
/
є
_state_var"
_generic_user_object
"
_generic_user_object
/
ѕ
_state_var"
_generic_user_object
"
_generic_user_object
/
і
_state_var"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

їtotal

јcount
љ	variables
њ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


ћtotal

ќcount
§
_fn_kwargs
ў	variables
џ	keras_api"П
_tf_keras_metricЄ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:	2Variable
:	2Variable
:	2Variable
:  (2total
:  (2count
0
ї0
ј1"
trackable_list_wrapper
.
љ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ћ0
ќ1"
trackable_list_wrapper
.
ў	variables"
_generic_user_object
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
.:,@@2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
.:,@@2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
.:,@@2Adam/conv2d_4/kernel/m
 :@2Adam/conv2d_4/bias/m
%:#
2Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	
2Adam/dense_1/kernel/m
:
2Adam/dense_1/bias/m
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
.:,@@2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
.:,@@2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
.:,@@2Adam/conv2d_4/kernel/v
 :@2Adam/conv2d_4/bias/v
%:#
2Adam/dense/kernel/v
:2Adam/dense/bias/v
&:$	
2Adam/dense_1/kernel/v
:
2Adam/dense_1/bias/v
ў2ћ
,__inference_sequential_layer_call_fn_1724296
,__inference_sequential_layer_call_fn_1723744
,__inference_sequential_layer_call_fn_1723833
,__inference_sequential_layer_call_fn_1724329Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ѕ2ђ
"__inference__wrapped_model_1722680Ы
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *;Ђ8
63
sequential_1_inputџџџџџџџџџ
ъ2ч
G__inference_sequential_layer_call_and_return_conditional_losses_1723646
G__inference_sequential_layer_call_and_return_conditional_losses_1724259
G__inference_sequential_layer_call_and_return_conditional_losses_1724194
G__inference_sequential_layer_call_and_return_conditional_losses_1723590Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
.__inference_sequential_1_layer_call_fn_1724597
.__inference_sequential_1_layer_call_fn_1723195
.__inference_sequential_1_layer_call_fn_1723204
.__inference_sequential_1_layer_call_fn_1724592Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
I__inference_sequential_1_layer_call_and_return_conditional_losses_1722935
I__inference_sequential_1_layer_call_and_return_conditional_losses_1724583
I__inference_sequential_1_layer_call_and_return_conditional_losses_1724579
I__inference_sequential_1_layer_call_and_return_conditional_losses_1722931Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
(__inference_conv2d_layer_call_fn_1724616Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_conv2d_layer_call_and_return_conditional_losses_1724607Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_activation_layer_call_fn_1724626Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_activation_layer_call_and_return_conditional_losses_1724621Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
/__inference_max_pooling2d_layer_call_fn_1723216р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
В2Џ
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1723210р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2d_1_layer_call_fn_1724645Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1724636Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_activation_1_layer_call_fn_1724655Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_activation_1_layer_call_and_return_conditional_losses_1724650Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_1_layer_call_fn_1723228р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1723222р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2d_2_layer_call_fn_1724674Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1724665Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_activation_2_layer_call_fn_1724684Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_activation_2_layer_call_and_return_conditional_losses_1724679Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_2_layer_call_fn_1723240р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1723234р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2d_3_layer_call_fn_1724703Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1724694Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_activation_3_layer_call_fn_1724713Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_activation_3_layer_call_and_return_conditional_losses_1724708Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_3_layer_call_fn_1723252р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1723246р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2d_4_layer_call_fn_1724732Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1724723Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_activation_4_layer_call_fn_1724742Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_activation_4_layer_call_and_return_conditional_losses_1724737Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_4_layer_call_fn_1723264р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1723258р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
)__inference_dropout_layer_call_fn_1724764
)__inference_dropout_layer_call_fn_1724769Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ц2У
D__inference_dropout_layer_call_and_return_conditional_losses_1724759
D__inference_dropout_layer_call_and_return_conditional_losses_1724754Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
г2а
)__inference_flatten_layer_call_fn_1724780Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_flatten_layer_call_and_return_conditional_losses_1724775Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_layer_call_fn_1724800Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_layer_call_and_return_conditional_losses_1724791Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_1_layer_call_fn_1724820Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_1_layer_call_and_return_conditional_losses_1724811Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
зBд
%__inference_signature_wrapper_1723876sequential_1_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Б
"__inference__wrapped_model_1722680$%23@ANO\]rsxyEЂB
;Ђ8
63
sequential_1_inputџџџџџџџџџ
Њ "1Њ.
,
dense_1!
dense_1џџџџџџџџџ
Е
I__inference_activation_1_layer_call_and_return_conditional_losses_1724650h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ}}@
Њ "-Ђ*
# 
0џџџџџџџџџ}}@
 
.__inference_activation_1_layer_call_fn_1724655[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ}}@
Њ " џџџџџџџџџ}}@Е
I__inference_activation_2_layer_call_and_return_conditional_losses_1724679h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ<<@
Њ "-Ђ*
# 
0џџџџџџџџџ<<@
 
.__inference_activation_2_layer_call_fn_1724684[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ<<@
Њ " џџџџџџџџџ<<@Е
I__inference_activation_3_layer_call_and_return_conditional_losses_1724708h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
.__inference_activation_3_layer_call_fn_1724713[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Е
I__inference_activation_4_layer_call_and_return_conditional_losses_1724737h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
.__inference_activation_4_layer_call_fn_1724742[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@З
G__inference_activation_layer_call_and_return_conditional_losses_1724621l9Ђ6
/Ђ,
*'
inputsџџџџџџџџџўў@
Њ "/Ђ,
%"
0џџџџџџџџџўў@
 
,__inference_activation_layer_call_fn_1724626_9Ђ6
/Ђ,
*'
inputsџџџџџџџџџўў@
Њ ""џџџџџџџџџўў@Е
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1724636l237Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ}}@
 
*__inference_conv2d_1_layer_call_fn_1724645_237Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ}}@Е
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1724665l@A7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ>>@
Њ "-Ђ*
# 
0џџџџџџџџџ<<@
 
*__inference_conv2d_2_layer_call_fn_1724674_@A7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ>>@
Њ " џџџџџџџџџ<<@Е
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1724694lNO7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
*__inference_conv2d_3_layer_call_fn_1724703_NO7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Е
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1724723l\]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
*__inference_conv2d_4_layer_call_fn_1724732_\]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@З
C__inference_conv2d_layer_call_and_return_conditional_losses_1724607p$%9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџўў@
 
(__inference_conv2d_layer_call_fn_1724616c$%9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџўў@Ѕ
D__inference_dense_1_layer_call_and_return_conditional_losses_1724811]xy0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ

 }
)__inference_dense_1_layer_call_fn_1724820Pxy0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
Є
B__inference_dense_layer_call_and_return_conditional_losses_1724791^rs0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 |
'__inference_dense_layer_call_fn_1724800Qrs0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџД
D__inference_dropout_layer_call_and_return_conditional_losses_1724754l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@
 Д
D__inference_dropout_layer_call_and_return_conditional_losses_1724759l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
)__inference_dropout_layer_call_fn_1724764_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ " џџџџџџџџџ@
)__inference_dropout_layer_call_fn_1724769_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ " џџџџџџџџџ@Љ
D__inference_flatten_layer_call_and_return_conditional_losses_1724775a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 
)__inference_flatten_layer_call_fn_1724780T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџя
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1723222RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_1_layer_call_fn_1723228RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1723234RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_2_layer_call_fn_1723240RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1723246RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_3_layer_call_fn_1723252RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1723258RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_4_layer_call_fn_1723264RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџэ
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1723210RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
/__inference_max_pooling2d_layer_call_fn_1723216RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџе
I__inference_sequential_1_layer_call_and_return_conditional_losses_1722931ѕіNЂK
DЂA
74
random_flip_1_inputџџџџџџџџџ
p

 
Њ "/Ђ,
%"
0џџџџџџџџџ
 Я
I__inference_sequential_1_layer_call_and_return_conditional_losses_1722935NЂK
DЂA
74
random_flip_1_inputџџџџџџџџџ
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџ
 Ч
I__inference_sequential_1_layer_call_and_return_conditional_losses_1724579zѕіAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ "/Ђ,
%"
0џџџџџџџџџ
 С
I__inference_sequential_1_layer_call_and_return_conditional_losses_1724583tAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџ
 Ќ
.__inference_sequential_1_layer_call_fn_1723195zѕіNЂK
DЂA
74
random_flip_1_inputџџџџџџџџџ
p

 
Њ ""џџџџџџџџџІ
.__inference_sequential_1_layer_call_fn_1723204tNЂK
DЂA
74
random_flip_1_inputџџџџџџџџџ
p 

 
Њ ""џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_1724592mѕіAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ ""џџџџџџџџџ
.__inference_sequential_1_layer_call_fn_1724597gAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ ""џџџџџџџџџж
G__inference_sequential_layer_call_and_return_conditional_losses_1723590ѕі$%23@ANO\]rsxyMЂJ
CЂ@
63
sequential_1_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 в
G__inference_sequential_layer_call_and_return_conditional_losses_1723646$%23@ANO\]rsxyMЂJ
CЂ@
63
sequential_1_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Щ
G__inference_sequential_layer_call_and_return_conditional_losses_1724194~ѕі$%23@ANO\]rsxyAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Х
G__inference_sequential_layer_call_and_return_conditional_losses_1724259z$%23@ANO\]rsxyAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 ­
,__inference_sequential_layer_call_fn_1723744}ѕі$%23@ANO\]rsxyMЂJ
CЂ@
63
sequential_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
Љ
,__inference_sequential_layer_call_fn_1723833y$%23@ANO\]rsxyMЂJ
CЂ@
63
sequential_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
Ё
,__inference_sequential_layer_call_fn_1724296qѕі$%23@ANO\]rsxyAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

,__inference_sequential_layer_call_fn_1724329m$%23@ANO\]rsxyAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
Ъ
%__inference_signature_wrapper_1723876 $%23@ANO\]rsxy[ЂX
Ђ 
QЊN
L
sequential_1_input63
sequential_1_inputџџџџџџџџџ"1Њ.
,
dense_1!
dense_1џџџџџџџџџ
