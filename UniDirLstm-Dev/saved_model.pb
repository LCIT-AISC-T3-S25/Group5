��
�&�&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
:
OnesLike
x"T
y"T"
Ttype:
2	

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
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
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
executor_typestring ��
^
StatelessRandomGetKeyCounter
seed"Tseed
key
counter"
Tseedtype0	:
2	
�
StatelessRandomUniformV2
shape"Tshape
key
counter
alg
output"dtype"
dtypetype0:
2"
Tshapetype0:
2	
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.17.02v2.17.0-rc1-2-gad6d8cc177d8�
�
%seed_generator_3/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_3/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_3/seed_generator_state
�
9seed_generator_3/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_3/seed_generator_state*
_output_shapes
:*
dtype0	
�
,sequential/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *=

debug_name/-sequential/batch_normalization_1/moving_mean/*
dtype0*
shape:@*=
shared_name.,sequential/batch_normalization_1/moving_mean
�
@sequential/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp,sequential/batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
�
.sequential/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *?

debug_name1/sequential/batch_normalization/moving_variance/*
dtype0*
shape:�*?
shared_name0.sequential/batch_normalization/moving_variance
�
Bsequential/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp.sequential/batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
�
%seed_generator_5/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_5/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_5/seed_generator_state
�
9seed_generator_5/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_5/seed_generator_state*
_output_shapes
:*
dtype0	
�
%seed_generator_6/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_6/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_6/seed_generator_state
�
9seed_generator_6/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_6/seed_generator_state*
_output_shapes
:*
dtype0	
�
0sequential/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *A

debug_name31sequential/batch_normalization_1/moving_variance/*
dtype0*
shape:@*A
shared_name20sequential/batch_normalization_1/moving_variance
�
Dsequential/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp0sequential/batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
�
*sequential/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *;

debug_name-+sequential/batch_normalization/moving_mean/*
dtype0*
shape:�*;
shared_name,*sequential/batch_normalization/moving_mean
�
>sequential/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp*sequential/batch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
�
%seed_generator_2/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_2/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_2/seed_generator_state
�
9seed_generator_2/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_2/seed_generator_state*
_output_shapes
:*
dtype0	
�
sequential/dense/kernelVarHandleOp*
_output_shapes
: *(

debug_namesequential/dense/kernel/*
dtype0*
shape:	�@*(
shared_namesequential/dense/kernel
�
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes
:	�@*
dtype0
�
#sequential/batch_normalization/betaVarHandleOp*
_output_shapes
: *4

debug_name&$sequential/batch_normalization/beta/*
dtype0*
shape:�*4
shared_name%#sequential/batch_normalization/beta
�
7sequential/batch_normalization/beta/Read/ReadVariableOpReadVariableOp#sequential/batch_normalization/beta*
_output_shapes	
:�*
dtype0
�
backward_lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: */

debug_name!backward_lstm/lstm_cell/kernel/*
dtype0*
shape:
��*/
shared_name backward_lstm/lstm_cell/kernel
�
2backward_lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpbackward_lstm/lstm_cell/kernel* 
_output_shapes
:
��*
dtype0
�
sequential/dense/biasVarHandleOp*
_output_shapes
: *&

debug_namesequential/dense/bias/*
dtype0*
shape:@*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:@*
dtype0
�
backward_lstm_1/lstm_cell/biasVarHandleOp*
_output_shapes
: */

debug_name!backward_lstm_1/lstm_cell/bias/*
dtype0*
shape:�*/
shared_name backward_lstm_1/lstm_cell/bias
�
2backward_lstm_1/lstm_cell/bias/Read/ReadVariableOpReadVariableOpbackward_lstm_1/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
forward_lstm_1/lstm_cell/kernelVarHandleOp*
_output_shapes
: *0

debug_name" forward_lstm_1/lstm_cell/kernel/*
dtype0*
shape:
��*0
shared_name!forward_lstm_1/lstm_cell/kernel
�
3forward_lstm_1/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpforward_lstm_1/lstm_cell/kernel* 
_output_shapes
:
��*
dtype0
�
*backward_lstm_1/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *;

debug_name-+backward_lstm_1/lstm_cell/recurrent_kernel/*
dtype0*
shape:	@�*;
shared_name,*backward_lstm_1/lstm_cell/recurrent_kernel
�
>backward_lstm_1/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp*backward_lstm_1/lstm_cell/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
forward_lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *.

debug_name forward_lstm/lstm_cell/kernel/*
dtype0*
shape:
��*.
shared_nameforward_lstm/lstm_cell/kernel
�
1forward_lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpforward_lstm/lstm_cell/kernel* 
_output_shapes
:
��*
dtype0
�
sequential/dense_1/kernelVarHandleOp*
_output_shapes
: **

debug_namesequential/dense_1/kernel/*
dtype0*
shape
:@**
shared_namesequential/dense_1/kernel
�
-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel*
_output_shapes

:@*
dtype0
�
)forward_lstm_1/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *:

debug_name,*forward_lstm_1/lstm_cell/recurrent_kernel/*
dtype0*
shape:	@�*:
shared_name+)forward_lstm_1/lstm_cell/recurrent_kernel
�
=forward_lstm_1/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)forward_lstm_1/lstm_cell/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
sequential/dense_1/biasVarHandleOp*
_output_shapes
: *(

debug_namesequential/dense_1/bias/*
dtype0*
shape:*(
shared_namesequential/dense_1/bias

+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes
:*
dtype0
�
%sequential/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *6

debug_name(&sequential/batch_normalization_1/beta/*
dtype0*
shape:@*6
shared_name'%sequential/batch_normalization_1/beta
�
9sequential/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp%sequential/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
�
(backward_lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *9

debug_name+)backward_lstm/lstm_cell/recurrent_kernel/*
dtype0*
shape:
��*9
shared_name*(backward_lstm/lstm_cell/recurrent_kernel
�
<backward_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp(backward_lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
forward_lstm_1/lstm_cell/biasVarHandleOp*
_output_shapes
: *.

debug_name forward_lstm_1/lstm_cell/bias/*
dtype0*
shape:�*.
shared_nameforward_lstm_1/lstm_cell/bias
�
1forward_lstm_1/lstm_cell/bias/Read/ReadVariableOpReadVariableOpforward_lstm_1/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
backward_lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *-

debug_namebackward_lstm/lstm_cell/bias/*
dtype0*
shape:�*-
shared_namebackward_lstm/lstm_cell/bias
�
0backward_lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpbackward_lstm/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
 backward_lstm_1/lstm_cell/kernelVarHandleOp*
_output_shapes
: *1

debug_name#!backward_lstm_1/lstm_cell/kernel/*
dtype0*
shape:
��*1
shared_name" backward_lstm_1/lstm_cell/kernel
�
4backward_lstm_1/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp backward_lstm_1/lstm_cell/kernel* 
_output_shapes
:
��*
dtype0
�
'forward_lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *8

debug_name*(forward_lstm/lstm_cell/recurrent_kernel/*
dtype0*
shape:
��*8
shared_name)'forward_lstm/lstm_cell/recurrent_kernel
�
;forward_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp'forward_lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
sequential/embedding/embeddingsVarHandleOp*
_output_shapes
: *0

debug_name" sequential/embedding/embeddings/*
dtype0*
shape:���*0
shared_name!sequential/embedding/embeddings
�
3sequential/embedding/embeddings/Read/ReadVariableOpReadVariableOpsequential/embedding/embeddings*!
_output_shapes
:���*
dtype0
�
&sequential/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *7

debug_name)'sequential/batch_normalization_1/gamma/*
dtype0*
shape:@*7
shared_name(&sequential/batch_normalization_1/gamma
�
:sequential/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp&sequential/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
�
$sequential/batch_normalization/gammaVarHandleOp*
_output_shapes
: *5

debug_name'%sequential/batch_normalization/gamma/*
dtype0*
shape:�*5
shared_name&$sequential/batch_normalization/gamma
�
8sequential/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp$sequential/batch_normalization/gamma*
_output_shapes	
:�*
dtype0
�
forward_lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *,

debug_nameforward_lstm/lstm_cell/bias/*
dtype0*
shape:�*,
shared_nameforward_lstm/lstm_cell/bias
�
/forward_lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpforward_lstm/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
sequential/dense_1/bias_1VarHandleOp*
_output_shapes
: **

debug_namesequential/dense_1/bias_1/*
dtype0*
shape:**
shared_namesequential/dense_1/bias_1
�
-sequential/dense_1/bias_1/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias_1*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpsequential/dense_1/bias_1*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
sequential/dense_1/kernel_1VarHandleOp*
_output_shapes
: *,

debug_namesequential/dense_1/kernel_1/*
dtype0*
shape
:@*,
shared_namesequential/dense_1/kernel_1
�
/sequential/dense_1/kernel_1/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel_1*
_output_shapes

:@*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpsequential/dense_1/kernel_1*
_class
loc:@Variable_1*
_output_shapes

:@*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:@*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:@*
dtype0
�
%seed_generator_7/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_7/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_7/seed_generator_state
�
9seed_generator_7/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_7/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_2/Initializer/ReadVariableOpReadVariableOp%seed_generator_7/seed_generator_state*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0	
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0	*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0	
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
�
2sequential/batch_normalization_1/moving_variance_1VarHandleOp*
_output_shapes
: *C

debug_name53sequential/batch_normalization_1/moving_variance_1/*
dtype0*
shape:@*C
shared_name42sequential/batch_normalization_1/moving_variance_1
�
Fsequential/batch_normalization_1/moving_variance_1/Read/ReadVariableOpReadVariableOp2sequential/batch_normalization_1/moving_variance_1*
_output_shapes
:@*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOp2sequential/batch_normalization_1/moving_variance_1*
_class
loc:@Variable_3*
_output_shapes
:@*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:@*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:@*
dtype0
�
.sequential/batch_normalization_1/moving_mean_1VarHandleOp*
_output_shapes
: *?

debug_name1/sequential/batch_normalization_1/moving_mean_1/*
dtype0*
shape:@*?
shared_name0.sequential/batch_normalization_1/moving_mean_1
�
Bsequential/batch_normalization_1/moving_mean_1/Read/ReadVariableOpReadVariableOp.sequential/batch_normalization_1/moving_mean_1*
_output_shapes
:@*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOp.sequential/batch_normalization_1/moving_mean_1*
_class
loc:@Variable_4*
_output_shapes
:@*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:@*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
e
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:@*
dtype0
�
'sequential/batch_normalization_1/beta_1VarHandleOp*
_output_shapes
: *8

debug_name*(sequential/batch_normalization_1/beta_1/*
dtype0*
shape:@*8
shared_name)'sequential/batch_normalization_1/beta_1
�
;sequential/batch_normalization_1/beta_1/Read/ReadVariableOpReadVariableOp'sequential/batch_normalization_1/beta_1*
_output_shapes
:@*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp'sequential/batch_normalization_1/beta_1*
_class
loc:@Variable_5*
_output_shapes
:@*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:@*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
e
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:@*
dtype0
�
(sequential/batch_normalization_1/gamma_1VarHandleOp*
_output_shapes
: *9

debug_name+)sequential/batch_normalization_1/gamma_1/*
dtype0*
shape:@*9
shared_name*(sequential/batch_normalization_1/gamma_1
�
<sequential/batch_normalization_1/gamma_1/Read/ReadVariableOpReadVariableOp(sequential/batch_normalization_1/gamma_1*
_output_shapes
:@*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOp(sequential/batch_normalization_1/gamma_1*
_class
loc:@Variable_6*
_output_shapes
:@*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:@*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:@*
dtype0
�
sequential/dense/bias_1VarHandleOp*
_output_shapes
: *(

debug_namesequential/dense/bias_1/*
dtype0*
shape:@*(
shared_namesequential/dense/bias_1

+sequential/dense/bias_1/Read/ReadVariableOpReadVariableOpsequential/dense/bias_1*
_output_shapes
:@*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpsequential/dense/bias_1*
_class
loc:@Variable_7*
_output_shapes
:@*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:@*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
e
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
:@*
dtype0
�
sequential/dense/kernel_1VarHandleOp*
_output_shapes
: **

debug_namesequential/dense/kernel_1/*
dtype0*
shape:	�@**
shared_namesequential/dense/kernel_1
�
-sequential/dense/kernel_1/Read/ReadVariableOpReadVariableOpsequential/dense/kernel_1*
_output_shapes
:	�@*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpsequential/dense/kernel_1*
_class
loc:@Variable_8*
_output_shapes
:	�@*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:	�@*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
j
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
:	�@*
dtype0
�
'seed_generator_5/seed_generator_state_1VarHandleOp*
_output_shapes
: *8

debug_name*(seed_generator_5/seed_generator_state_1/*
dtype0	*
shape:*8
shared_name)'seed_generator_5/seed_generator_state_1
�
;seed_generator_5/seed_generator_state_1/Read/ReadVariableOpReadVariableOp'seed_generator_5/seed_generator_state_1*
_output_shapes
:*
dtype0	
�
%Variable_9/Initializer/ReadVariableOpReadVariableOp'seed_generator_5/seed_generator_state_1*
_class
loc:@Variable_9*
_output_shapes
:*
dtype0	
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0	*
shape:*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0	
e
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
:*
dtype0	
�
 backward_lstm_1/lstm_cell/bias_1VarHandleOp*
_output_shapes
: *1

debug_name#!backward_lstm_1/lstm_cell/bias_1/*
dtype0*
shape:�*1
shared_name" backward_lstm_1/lstm_cell/bias_1
�
4backward_lstm_1/lstm_cell/bias_1/Read/ReadVariableOpReadVariableOp backward_lstm_1/lstm_cell/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOp backward_lstm_1/lstm_cell/bias_1*
_class
loc:@Variable_10*
_output_shapes	
:�*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:�*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
h
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes	
:�*
dtype0
�
,backward_lstm_1/lstm_cell/recurrent_kernel_1VarHandleOp*
_output_shapes
: *=

debug_name/-backward_lstm_1/lstm_cell/recurrent_kernel_1/*
dtype0*
shape:	@�*=
shared_name.,backward_lstm_1/lstm_cell/recurrent_kernel_1
�
@backward_lstm_1/lstm_cell/recurrent_kernel_1/Read/ReadVariableOpReadVariableOp,backward_lstm_1/lstm_cell/recurrent_kernel_1*
_output_shapes
:	@�*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOp,backward_lstm_1/lstm_cell/recurrent_kernel_1*
_class
loc:@Variable_11*
_output_shapes
:	@�*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:	@�*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
l
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
:	@�*
dtype0
�
"backward_lstm_1/lstm_cell/kernel_1VarHandleOp*
_output_shapes
: *3

debug_name%#backward_lstm_1/lstm_cell/kernel_1/*
dtype0*
shape:
��*3
shared_name$"backward_lstm_1/lstm_cell/kernel_1
�
6backward_lstm_1/lstm_cell/kernel_1/Read/ReadVariableOpReadVariableOp"backward_lstm_1/lstm_cell/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOp"backward_lstm_1/lstm_cell/kernel_1*
_class
loc:@Variable_12* 
_output_shapes
:
��*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:
��*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
m
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12* 
_output_shapes
:
��*
dtype0
�
'seed_generator_6/seed_generator_state_1VarHandleOp*
_output_shapes
: *8

debug_name*(seed_generator_6/seed_generator_state_1/*
dtype0	*
shape:*8
shared_name)'seed_generator_6/seed_generator_state_1
�
;seed_generator_6/seed_generator_state_1/Read/ReadVariableOpReadVariableOp'seed_generator_6/seed_generator_state_1*
_output_shapes
:*
dtype0	
�
&Variable_13/Initializer/ReadVariableOpReadVariableOp'seed_generator_6/seed_generator_state_1*
_class
loc:@Variable_13*
_output_shapes
:*
dtype0	
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0	*
shape:*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0	
g
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes
:*
dtype0	
�
forward_lstm_1/lstm_cell/bias_1VarHandleOp*
_output_shapes
: *0

debug_name" forward_lstm_1/lstm_cell/bias_1/*
dtype0*
shape:�*0
shared_name!forward_lstm_1/lstm_cell/bias_1
�
3forward_lstm_1/lstm_cell/bias_1/Read/ReadVariableOpReadVariableOpforward_lstm_1/lstm_cell/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOpforward_lstm_1/lstm_cell/bias_1*
_class
loc:@Variable_14*
_output_shapes	
:�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
h
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes	
:�*
dtype0
�
+forward_lstm_1/lstm_cell/recurrent_kernel_1VarHandleOp*
_output_shapes
: *<

debug_name.,forward_lstm_1/lstm_cell/recurrent_kernel_1/*
dtype0*
shape:	@�*<
shared_name-+forward_lstm_1/lstm_cell/recurrent_kernel_1
�
?forward_lstm_1/lstm_cell/recurrent_kernel_1/Read/ReadVariableOpReadVariableOp+forward_lstm_1/lstm_cell/recurrent_kernel_1*
_output_shapes
:	@�*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOp+forward_lstm_1/lstm_cell/recurrent_kernel_1*
_class
loc:@Variable_15*
_output_shapes
:	@�*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:	@�*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
l
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes
:	@�*
dtype0
�
!forward_lstm_1/lstm_cell/kernel_1VarHandleOp*
_output_shapes
: *2

debug_name$"forward_lstm_1/lstm_cell/kernel_1/*
dtype0*
shape:
��*2
shared_name#!forward_lstm_1/lstm_cell/kernel_1
�
5forward_lstm_1/lstm_cell/kernel_1/Read/ReadVariableOpReadVariableOp!forward_lstm_1/lstm_cell/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOp!forward_lstm_1/lstm_cell/kernel_1*
_class
loc:@Variable_16* 
_output_shapes
:
��*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:
��*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
m
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16* 
_output_shapes
:
��*
dtype0
�
0sequential/batch_normalization/moving_variance_1VarHandleOp*
_output_shapes
: *A

debug_name31sequential/batch_normalization/moving_variance_1/*
dtype0*
shape:�*A
shared_name20sequential/batch_normalization/moving_variance_1
�
Dsequential/batch_normalization/moving_variance_1/Read/ReadVariableOpReadVariableOp0sequential/batch_normalization/moving_variance_1*
_output_shapes	
:�*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOp0sequential/batch_normalization/moving_variance_1*
_class
loc:@Variable_17*
_output_shapes	
:�*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:�*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
h
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes	
:�*
dtype0
�
,sequential/batch_normalization/moving_mean_1VarHandleOp*
_output_shapes
: *=

debug_name/-sequential/batch_normalization/moving_mean_1/*
dtype0*
shape:�*=
shared_name.,sequential/batch_normalization/moving_mean_1
�
@sequential/batch_normalization/moving_mean_1/Read/ReadVariableOpReadVariableOp,sequential/batch_normalization/moving_mean_1*
_output_shapes	
:�*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOp,sequential/batch_normalization/moving_mean_1*
_class
loc:@Variable_18*
_output_shapes	
:�*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape:�*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
h
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes	
:�*
dtype0
�
%sequential/batch_normalization/beta_1VarHandleOp*
_output_shapes
: *6

debug_name(&sequential/batch_normalization/beta_1/*
dtype0*
shape:�*6
shared_name'%sequential/batch_normalization/beta_1
�
9sequential/batch_normalization/beta_1/Read/ReadVariableOpReadVariableOp%sequential/batch_normalization/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOp%sequential/batch_normalization/beta_1*
_class
loc:@Variable_19*
_output_shapes	
:�*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:�*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
h
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*
_output_shapes	
:�*
dtype0
�
&sequential/batch_normalization/gamma_1VarHandleOp*
_output_shapes
: *7

debug_name)'sequential/batch_normalization/gamma_1/*
dtype0*
shape:�*7
shared_name(&sequential/batch_normalization/gamma_1
�
:sequential/batch_normalization/gamma_1/Read/ReadVariableOpReadVariableOp&sequential/batch_normalization/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOp&sequential/batch_normalization/gamma_1*
_class
loc:@Variable_20*
_output_shapes	
:�*
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape:�*
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
h
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes	
:�*
dtype0
�
'seed_generator_2/seed_generator_state_1VarHandleOp*
_output_shapes
: *8

debug_name*(seed_generator_2/seed_generator_state_1/*
dtype0	*
shape:*8
shared_name)'seed_generator_2/seed_generator_state_1
�
;seed_generator_2/seed_generator_state_1/Read/ReadVariableOpReadVariableOp'seed_generator_2/seed_generator_state_1*
_output_shapes
:*
dtype0	
�
&Variable_21/Initializer/ReadVariableOpReadVariableOp'seed_generator_2/seed_generator_state_1*
_class
loc:@Variable_21*
_output_shapes
:*
dtype0	
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0	*
shape:*
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0	
g
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*
_output_shapes
:*
dtype0	
�
backward_lstm/lstm_cell/bias_1VarHandleOp*
_output_shapes
: */

debug_name!backward_lstm/lstm_cell/bias_1/*
dtype0*
shape:�*/
shared_name backward_lstm/lstm_cell/bias_1
�
2backward_lstm/lstm_cell/bias_1/Read/ReadVariableOpReadVariableOpbackward_lstm/lstm_cell/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOpbackward_lstm/lstm_cell/bias_1*
_class
loc:@Variable_22*
_output_shapes	
:�*
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape:�*
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
h
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*
_output_shapes	
:�*
dtype0
�
*backward_lstm/lstm_cell/recurrent_kernel_1VarHandleOp*
_output_shapes
: *;

debug_name-+backward_lstm/lstm_cell/recurrent_kernel_1/*
dtype0*
shape:
��*;
shared_name,*backward_lstm/lstm_cell/recurrent_kernel_1
�
>backward_lstm/lstm_cell/recurrent_kernel_1/Read/ReadVariableOpReadVariableOp*backward_lstm/lstm_cell/recurrent_kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOp*backward_lstm/lstm_cell/recurrent_kernel_1*
_class
loc:@Variable_23* 
_output_shapes
:
��*
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape:
��*
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
m
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23* 
_output_shapes
:
��*
dtype0
�
 backward_lstm/lstm_cell/kernel_1VarHandleOp*
_output_shapes
: *1

debug_name#!backward_lstm/lstm_cell/kernel_1/*
dtype0*
shape:
��*1
shared_name" backward_lstm/lstm_cell/kernel_1
�
4backward_lstm/lstm_cell/kernel_1/Read/ReadVariableOpReadVariableOp backward_lstm/lstm_cell/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOp backward_lstm/lstm_cell/kernel_1*
_class
loc:@Variable_24* 
_output_shapes
:
��*
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape:
��*
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
m
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24* 
_output_shapes
:
��*
dtype0
�
'seed_generator_3/seed_generator_state_1VarHandleOp*
_output_shapes
: *8

debug_name*(seed_generator_3/seed_generator_state_1/*
dtype0	*
shape:*8
shared_name)'seed_generator_3/seed_generator_state_1
�
;seed_generator_3/seed_generator_state_1/Read/ReadVariableOpReadVariableOp'seed_generator_3/seed_generator_state_1*
_output_shapes
:*
dtype0	
�
&Variable_25/Initializer/ReadVariableOpReadVariableOp'seed_generator_3/seed_generator_state_1*
_class
loc:@Variable_25*
_output_shapes
:*
dtype0	
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0	*
shape:*
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0	
g
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*
_output_shapes
:*
dtype0	
�
forward_lstm/lstm_cell/bias_1VarHandleOp*
_output_shapes
: *.

debug_name forward_lstm/lstm_cell/bias_1/*
dtype0*
shape:�*.
shared_nameforward_lstm/lstm_cell/bias_1
�
1forward_lstm/lstm_cell/bias_1/Read/ReadVariableOpReadVariableOpforward_lstm/lstm_cell/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_26/Initializer/ReadVariableOpReadVariableOpforward_lstm/lstm_cell/bias_1*
_class
loc:@Variable_26*
_output_shapes	
:�*
dtype0
�
Variable_26VarHandleOp*
_class
loc:@Variable_26*
_output_shapes
: *

debug_nameVariable_26/*
dtype0*
shape:�*
shared_nameVariable_26
g
,Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_26*
_output_shapes
: 
h
Variable_26/AssignAssignVariableOpVariable_26&Variable_26/Initializer/ReadVariableOp*
dtype0
h
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26*
_output_shapes	
:�*
dtype0
�
)forward_lstm/lstm_cell/recurrent_kernel_1VarHandleOp*
_output_shapes
: *:

debug_name,*forward_lstm/lstm_cell/recurrent_kernel_1/*
dtype0*
shape:
��*:
shared_name+)forward_lstm/lstm_cell/recurrent_kernel_1
�
=forward_lstm/lstm_cell/recurrent_kernel_1/Read/ReadVariableOpReadVariableOp)forward_lstm/lstm_cell/recurrent_kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_27/Initializer/ReadVariableOpReadVariableOp)forward_lstm/lstm_cell/recurrent_kernel_1*
_class
loc:@Variable_27* 
_output_shapes
:
��*
dtype0
�
Variable_27VarHandleOp*
_class
loc:@Variable_27*
_output_shapes
: *

debug_nameVariable_27/*
dtype0*
shape:
��*
shared_nameVariable_27
g
,Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_27*
_output_shapes
: 
h
Variable_27/AssignAssignVariableOpVariable_27&Variable_27/Initializer/ReadVariableOp*
dtype0
m
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27* 
_output_shapes
:
��*
dtype0
�
forward_lstm/lstm_cell/kernel_1VarHandleOp*
_output_shapes
: *0

debug_name" forward_lstm/lstm_cell/kernel_1/*
dtype0*
shape:
��*0
shared_name!forward_lstm/lstm_cell/kernel_1
�
3forward_lstm/lstm_cell/kernel_1/Read/ReadVariableOpReadVariableOpforward_lstm/lstm_cell/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_28/Initializer/ReadVariableOpReadVariableOpforward_lstm/lstm_cell/kernel_1*
_class
loc:@Variable_28* 
_output_shapes
:
��*
dtype0
�
Variable_28VarHandleOp*
_class
loc:@Variable_28*
_output_shapes
: *

debug_nameVariable_28/*
dtype0*
shape:
��*
shared_nameVariable_28
g
,Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_28*
_output_shapes
: 
h
Variable_28/AssignAssignVariableOpVariable_28&Variable_28/Initializer/ReadVariableOp*
dtype0
m
Variable_28/Read/ReadVariableOpReadVariableOpVariable_28* 
_output_shapes
:
��*
dtype0
�
#seed_generator/seed_generator_stateVarHandleOp*
_output_shapes
: *4

debug_name&$seed_generator/seed_generator_state/*
dtype0	*
shape:*4
shared_name%#seed_generator/seed_generator_state
�
7seed_generator/seed_generator_state/Read/ReadVariableOpReadVariableOp#seed_generator/seed_generator_state*
_output_shapes
:*
dtype0	
�
&Variable_29/Initializer/ReadVariableOpReadVariableOp#seed_generator/seed_generator_state*
_class
loc:@Variable_29*
_output_shapes
:*
dtype0	
�
Variable_29VarHandleOp*
_class
loc:@Variable_29*
_output_shapes
: *

debug_nameVariable_29/*
dtype0	*
shape:*
shared_nameVariable_29
g
,Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_29*
_output_shapes
: 
h
Variable_29/AssignAssignVariableOpVariable_29&Variable_29/Initializer/ReadVariableOp*
dtype0	
g
Variable_29/Read/ReadVariableOpReadVariableOpVariable_29*
_output_shapes
:*
dtype0	
�
!sequential/embedding/embeddings_1VarHandleOp*
_output_shapes
: *2

debug_name$"sequential/embedding/embeddings_1/*
dtype0*
shape:���*2
shared_name#!sequential/embedding/embeddings_1
�
5sequential/embedding/embeddings_1/Read/ReadVariableOpReadVariableOp!sequential/embedding/embeddings_1*!
_output_shapes
:���*
dtype0
�
&Variable_30/Initializer/ReadVariableOpReadVariableOp!sequential/embedding/embeddings_1*
_class
loc:@Variable_30*!
_output_shapes
:���*
dtype0
�
Variable_30VarHandleOp*
_class
loc:@Variable_30*
_output_shapes
: *

debug_nameVariable_30/*
dtype0*
shape:���*
shared_nameVariable_30
g
,Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_30*
_output_shapes
: 
h
Variable_30/AssignAssignVariableOpVariable_30&Variable_30/Initializer/ReadVariableOp*
dtype0
n
Variable_30/Read/ReadVariableOpReadVariableOpVariable_30*!
_output_shapes
:���*
dtype0
t
serve_input_layerPlaceholder*'
_output_shapes
:���������x*
dtype0*
shape:���������x
�
StatefulPartitionedCallStatefulPartitionedCallserve_input_layer!sequential/embedding/embeddings_1'seed_generator_3/seed_generator_state_1forward_lstm/lstm_cell/kernel_1)forward_lstm/lstm_cell/recurrent_kernel_1forward_lstm/lstm_cell/bias_1'seed_generator_2/seed_generator_state_1 backward_lstm/lstm_cell/kernel_1*backward_lstm/lstm_cell/recurrent_kernel_1backward_lstm/lstm_cell/bias_1,sequential/batch_normalization/moving_mean_10sequential/batch_normalization/moving_variance_1&sequential/batch_normalization/gamma_1%sequential/batch_normalization/beta_1'seed_generator_6/seed_generator_state_1!forward_lstm_1/lstm_cell/kernel_1+forward_lstm_1/lstm_cell/recurrent_kernel_1forward_lstm_1/lstm_cell/bias_1'seed_generator_5/seed_generator_state_1"backward_lstm_1/lstm_cell/kernel_1,backward_lstm_1/lstm_cell/recurrent_kernel_1 backward_lstm_1/lstm_cell/bias_1sequential/dense/kernel_1sequential/dense/bias_1.sequential/batch_normalization_1/moving_mean_12sequential/batch_normalization_1/moving_variance_1(sequential/batch_normalization_1/gamma_1'sequential/batch_normalization_1/beta_1sequential/dense_1/kernel_1sequential/dense_1/bias_1*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_signature_wrapper___call___2565
~
serving_default_input_layerPlaceholder*'
_output_shapes
:���������x*
dtype0*
shape:���������x
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_layer!sequential/embedding/embeddings_1'seed_generator_3/seed_generator_state_1forward_lstm/lstm_cell/kernel_1)forward_lstm/lstm_cell/recurrent_kernel_1forward_lstm/lstm_cell/bias_1'seed_generator_2/seed_generator_state_1 backward_lstm/lstm_cell/kernel_1*backward_lstm/lstm_cell/recurrent_kernel_1backward_lstm/lstm_cell/bias_1,sequential/batch_normalization/moving_mean_10sequential/batch_normalization/moving_variance_1&sequential/batch_normalization/gamma_1%sequential/batch_normalization/beta_1'seed_generator_6/seed_generator_state_1!forward_lstm_1/lstm_cell/kernel_1+forward_lstm_1/lstm_cell/recurrent_kernel_1forward_lstm_1/lstm_cell/bias_1'seed_generator_5/seed_generator_state_1"backward_lstm_1/lstm_cell/kernel_1,backward_lstm_1/lstm_cell/recurrent_kernel_1 backward_lstm_1/lstm_cell/bias_1sequential/dense/kernel_1sequential/dense/bias_1.sequential/batch_normalization_1/moving_mean_12sequential/batch_normalization_1/moving_variance_1(sequential/batch_normalization_1/gamma_1'sequential/batch_normalization_1/beta_1sequential/dense_1/kernel_1sequential/dense_1/bias_1*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_signature_wrapper___call___2628

NoOpNoOp
�3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�2
value�2B�2 B�2
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30*
�
0

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
%19
&20*
J
	0
1
2
3
4
5
6
"7
#8
$9*
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
@25
A26
B27
C28*
* 

Dtrace_0* 
"
	Eserve
Fserving_default* 
KE
VARIABLE_VALUEVariable_30&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_29&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_28&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_27&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_26&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_25&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_24&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_23&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_22&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_21&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_20'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_19'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_18'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_17'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_16'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_15'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_14'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_13'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_12'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_11'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_10'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_9'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_8'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_7'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_6'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_5'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_4'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_3'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_2'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_1'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEVariable'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEforward_lstm/lstm_cell/bias_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE&sequential/batch_normalization/gamma_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE(sequential/batch_normalization_1/gamma_1+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE!sequential/embedding/embeddings_1+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE)forward_lstm/lstm_cell/recurrent_kernel_1+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE"backward_lstm_1/lstm_cell/kernel_1+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEbackward_lstm/lstm_cell/bias_1+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEforward_lstm_1/lstm_cell/bias_1+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE*backward_lstm/lstm_cell/recurrent_kernel_1+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE'sequential/batch_normalization_1/beta_1+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsequential/dense_1/bias_1,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE+forward_lstm_1/lstm_cell/recurrent_kernel_1,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEsequential/dense_1/kernel_1,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEforward_lstm/lstm_cell/kernel_1,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE,backward_lstm_1/lstm_cell/recurrent_kernel_1,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE!forward_lstm_1/lstm_cell/kernel_1,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE backward_lstm_1/lstm_cell/bias_1,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEsequential/dense/bias_1,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE backward_lstm/lstm_cell/kernel_1,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE%sequential/batch_normalization/beta_1,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsequential/dense/kernel_1,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE'seed_generator_2/seed_generator_state_1,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE,sequential/batch_normalization/moving_mean_1,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE2sequential/batch_normalization_1/moving_variance_1,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE'seed_generator_6/seed_generator_state_1,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE'seed_generator_5/seed_generator_state_1,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE0sequential/batch_normalization/moving_variance_1,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE.sequential/batch_normalization_1/moving_mean_1,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE'seed_generator_3/seed_generator_state_1,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variableforward_lstm/lstm_cell/bias_1&sequential/batch_normalization/gamma_1(sequential/batch_normalization_1/gamma_1!sequential/embedding/embeddings_1)forward_lstm/lstm_cell/recurrent_kernel_1"backward_lstm_1/lstm_cell/kernel_1backward_lstm/lstm_cell/bias_1forward_lstm_1/lstm_cell/bias_1*backward_lstm/lstm_cell/recurrent_kernel_1'sequential/batch_normalization_1/beta_1sequential/dense_1/bias_1+forward_lstm_1/lstm_cell/recurrent_kernel_1sequential/dense_1/kernel_1forward_lstm/lstm_cell/kernel_1,backward_lstm_1/lstm_cell/recurrent_kernel_1!forward_lstm_1/lstm_cell/kernel_1 backward_lstm_1/lstm_cell/bias_1sequential/dense/bias_1 backward_lstm/lstm_cell/kernel_1%sequential/batch_normalization/beta_1sequential/dense/kernel_1'seed_generator_2/seed_generator_state_1,sequential/batch_normalization/moving_mean_12sequential/batch_normalization_1/moving_variance_1'seed_generator_6/seed_generator_state_1'seed_generator_5/seed_generator_state_10sequential/batch_normalization/moving_variance_1.sequential/batch_normalization_1/moving_mean_1'seed_generator_3/seed_generator_state_1Const*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_3136
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variableforward_lstm/lstm_cell/bias_1&sequential/batch_normalization/gamma_1(sequential/batch_normalization_1/gamma_1!sequential/embedding/embeddings_1)forward_lstm/lstm_cell/recurrent_kernel_1"backward_lstm_1/lstm_cell/kernel_1backward_lstm/lstm_cell/bias_1forward_lstm_1/lstm_cell/bias_1*backward_lstm/lstm_cell/recurrent_kernel_1'sequential/batch_normalization_1/beta_1sequential/dense_1/bias_1+forward_lstm_1/lstm_cell/recurrent_kernel_1sequential/dense_1/kernel_1forward_lstm/lstm_cell/kernel_1,backward_lstm_1/lstm_cell/recurrent_kernel_1!forward_lstm_1/lstm_cell/kernel_1 backward_lstm_1/lstm_cell/bias_1sequential/dense/bias_1 backward_lstm/lstm_cell/kernel_1%sequential/batch_normalization/beta_1sequential/dense/kernel_1'seed_generator_2/seed_generator_state_1,sequential/batch_normalization/moving_mean_12sequential/batch_normalization_1/moving_variance_1'seed_generator_6/seed_generator_state_1'seed_generator_5/seed_generator_state_10sequential/batch_normalization/moving_variance_1.sequential/batch_normalization_1/moving_mean_1'seed_generator_3/seed_generator_state_1*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_3325��
��	
�0
__inference___call___2501
input_layerK
6sequential_1_embedding_1_shape_readvariableop_resource:���Q
Csequential_1_bidirectional_1_forward_lstm_1_readvariableop_resource:	h
Tsequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_cast_readvariableop_resource:
��j
Vsequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_cast_1_readvariableop_resource:
��d
Usequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_add_1_readvariableop_resource:	�R
Dsequential_1_bidirectional_1_backward_lstm_1_readvariableop_resource:	i
Usequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_cast_readvariableop_resource:
��k
Wsequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_cast_1_readvariableop_resource:
��e
Vsequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_add_1_readvariableop_resource:	�N
?sequential_1_batch_normalization_1_cast_readvariableop_resource:	�P
Asequential_1_batch_normalization_1_cast_1_readvariableop_resource:	�P
Asequential_1_batch_normalization_1_cast_2_readvariableop_resource:	�P
Asequential_1_batch_normalization_1_cast_3_readvariableop_resource:	�U
Gsequential_1_bidirectional_1_2_forward_lstm_1_1_readvariableop_resource:	l
Xsequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_cast_readvariableop_resource:
��m
Zsequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_cast_1_readvariableop_resource:	@�h
Ysequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_add_1_readvariableop_resource:	�V
Hsequential_1_bidirectional_1_2_backward_lstm_1_1_readvariableop_resource:	m
Ysequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_cast_readvariableop_resource:
��n
[sequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_cast_1_readvariableop_resource:	@�i
Zsequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_add_1_readvariableop_resource:	�D
1sequential_1_dense_1_cast_readvariableop_resource:	�@>
0sequential_1_dense_1_add_readvariableop_resource:@O
Asequential_1_batch_normalization_1_2_cast_readvariableop_resource:@Q
Csequential_1_batch_normalization_1_2_cast_1_readvariableop_resource:@Q
Csequential_1_batch_normalization_1_2_cast_2_readvariableop_resource:@Q
Csequential_1_batch_normalization_1_2_cast_3_readvariableop_resource:@E
3sequential_1_dense_1_2_cast_readvariableop_resource:@@
2sequential_1_dense_1_2_add_readvariableop_resource:
identity��6sequential_1/batch_normalization_1/Cast/ReadVariableOp�8sequential_1/batch_normalization_1/Cast_1/ReadVariableOp�8sequential_1/batch_normalization_1/Cast_2/ReadVariableOp�8sequential_1/batch_normalization_1/Cast_3/ReadVariableOp�8sequential_1/batch_normalization_1_2/Cast/ReadVariableOp�:sequential_1/batch_normalization_1_2/Cast_1/ReadVariableOp�:sequential_1/batch_normalization_1_2/Cast_2/ReadVariableOp�:sequential_1/batch_normalization_1_2/Cast_3/ReadVariableOp�?sequential_1/bidirectional_1/backward_lstm_1/Add/ReadVariableOp�Asequential_1/bidirectional_1/backward_lstm_1/Add_1/ReadVariableOp�=sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp�?sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp_1�;sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp�=sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp_1�Lsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast/ReadVariableOp�Nsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOp�Msequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_1/ReadVariableOp�2sequential_1/bidirectional_1/backward_lstm_1/while�>sequential_1/bidirectional_1/forward_lstm_1/Add/ReadVariableOp�@sequential_1/bidirectional_1/forward_lstm_1/Add_1/ReadVariableOp�<sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp�>sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp_1�:sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp�<sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp_1�Ksequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast/ReadVariableOp�Msequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOp�Lsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_1/ReadVariableOp�1sequential_1/bidirectional_1/forward_lstm_1/while�Csequential_1/bidirectional_1_2/backward_lstm_1_1/Add/ReadVariableOp�Esequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1/ReadVariableOp�Asequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp�Csequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp_1�?sequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp�Asequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp_1�Psequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOp�Rsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOp�Qsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOp�6sequential_1/bidirectional_1_2/backward_lstm_1_1/while�Bsequential_1/bidirectional_1_2/forward_lstm_1_1/Add/ReadVariableOp�Dsequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1/ReadVariableOp�@sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp�Bsequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp_1�>sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp�@sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp_1�Osequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOp�Qsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOp�Psequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOp�5sequential_1/bidirectional_1_2/forward_lstm_1_1/while�'sequential_1/dense_1/Add/ReadVariableOp�(sequential_1/dense_1/Cast/ReadVariableOp�)sequential_1/dense_1_2/Add/ReadVariableOp�*sequential_1/dense_1_2/Cast/ReadVariableOp�0sequential_1/embedding_1/GatherV2/ReadVariableOps
sequential_1/embedding_1/CastCastinput_layer*

DstT0*

SrcT0*'
_output_shapes
:���������xa
sequential_1/embedding_1/Less/yConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_1/embedding_1/LessLess!sequential_1/embedding_1/Cast:y:0(sequential_1/embedding_1/Less/y:output:0*
T0*'
_output_shapes
:���������x�
-sequential_1/embedding_1/Shape/ReadVariableOpReadVariableOp6sequential_1_embedding_1_shape_readvariableop_resource*!
_output_shapes
:���*
dtype0o
sequential_1/embedding_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"0u  �   v
,sequential_1/embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_1/embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_1/embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_1/embedding_1/strided_sliceStridedSlice'sequential_1/embedding_1/Shape:output:05sequential_1/embedding_1/strided_slice/stack:output:07sequential_1/embedding_1/strided_slice/stack_1:output:07sequential_1/embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
sequential_1/embedding_1/addAddV2!sequential_1/embedding_1/Cast:y:0/sequential_1/embedding_1/strided_slice:output:0*
T0*'
_output_shapes
:���������x�
!sequential_1/embedding_1/SelectV2SelectV2!sequential_1/embedding_1/Less:z:0 sequential_1/embedding_1/add:z:0!sequential_1/embedding_1/Cast:y:0*
T0*'
_output_shapes
:���������x�
0sequential_1/embedding_1/GatherV2/ReadVariableOpReadVariableOp6sequential_1_embedding_1_shape_readvariableop_resource*!
_output_shapes
:���*
dtype0h
&sequential_1/embedding_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential_1/embedding_1/GatherV2GatherV28sequential_1/embedding_1/GatherV2/ReadVariableOp:value:0*sequential_1/embedding_1/SelectV2:output:0/sequential_1/embedding_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:���������x�T
sequential_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : f
sequential_1/CastCastsequential_1/Const:output:0*

DstT0*

SrcT0*
_output_shapes
: w
sequential_1/NotEqualNotEqualinput_layersequential_1/Cast:y:0*
T0*'
_output_shapes
:���������xV
sequential_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B : j
sequential_1/Cast_1Castsequential_1/Const_1:output:0*

DstT0*

SrcT0*
_output_shapes
: {
sequential_1/NotEqual_1NotEqualinput_layersequential_1/Cast_1:y:0*
T0*'
_output_shapes
:���������x�
1sequential_1/bidirectional_1/forward_lstm_1/ShapeShape*sequential_1/embedding_1/GatherV2:output:0*
T0*
_output_shapes
::���
?sequential_1/bidirectional_1/forward_lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Asequential_1/bidirectional_1/forward_lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Asequential_1/bidirectional_1/forward_lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential_1/bidirectional_1/forward_lstm_1/strided_sliceStridedSlice:sequential_1/bidirectional_1/forward_lstm_1/Shape:output:0Hsequential_1/bidirectional_1/forward_lstm_1/strided_slice/stack:output:0Jsequential_1/bidirectional_1/forward_lstm_1/strided_slice/stack_1:output:0Jsequential_1/bidirectional_1/forward_lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
:sequential_1/bidirectional_1/forward_lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
8sequential_1/bidirectional_1/forward_lstm_1/zeros/packedPackBsequential_1/bidirectional_1/forward_lstm_1/strided_slice:output:0Csequential_1/bidirectional_1/forward_lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:|
7sequential_1/bidirectional_1/forward_lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
1sequential_1/bidirectional_1/forward_lstm_1/zerosFillAsequential_1/bidirectional_1/forward_lstm_1/zeros/packed:output:0@sequential_1/bidirectional_1/forward_lstm_1/zeros/Const:output:0*
T0*(
_output_shapes
:����������
<sequential_1/bidirectional_1/forward_lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
:sequential_1/bidirectional_1/forward_lstm_1/zeros_1/packedPackBsequential_1/bidirectional_1/forward_lstm_1/strided_slice:output:0Esequential_1/bidirectional_1/forward_lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:~
9sequential_1/bidirectional_1/forward_lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
3sequential_1/bidirectional_1/forward_lstm_1/zeros_1FillCsequential_1/bidirectional_1/forward_lstm_1/zeros_1/packed:output:0Bsequential_1/bidirectional_1/forward_lstm_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:�����������
Asequential_1/bidirectional_1/forward_lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
Csequential_1/bidirectional_1/forward_lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
Csequential_1/bidirectional_1/forward_lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
;sequential_1/bidirectional_1/forward_lstm_1/strided_slice_1StridedSlice*sequential_1/embedding_1/GatherV2:output:0Jsequential_1/bidirectional_1/forward_lstm_1/strided_slice_1/stack:output:0Lsequential_1/bidirectional_1/forward_lstm_1/strided_slice_1/stack_1:output:0Lsequential_1/bidirectional_1/forward_lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
5sequential_1/bidirectional_1/forward_lstm_1/ones_likeOnesLikeDsequential_1/bidirectional_1/forward_lstm_1/strided_slice_1:output:0*
T0*(
_output_shapes
:�����������
:sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOpReadVariableOpCsequential_1_bidirectional_1_forward_lstm_1_readvariableop_resource*
_output_shapes
:*
dtype0	s
1sequential_1/bidirectional_1/forward_lstm_1/mul/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
/sequential_1/bidirectional_1/forward_lstm_1/mulMulBsequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp:value:0:sequential_1/bidirectional_1/forward_lstm_1/mul/y:output:0*
T0	*
_output_shapes
:�
1sequential_1/bidirectional_1/forward_lstm_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"               �
>sequential_1/bidirectional_1/forward_lstm_1/Add/ReadVariableOpReadVariableOpCsequential_1_bidirectional_1_forward_lstm_1_readvariableop_resource*
_output_shapes
:*
dtype0	�
/sequential_1/bidirectional_1/forward_lstm_1/AddAddV2Fsequential_1/bidirectional_1/forward_lstm_1/Add/ReadVariableOp:value:0:sequential_1/bidirectional_1/forward_lstm_1/Const:output:0*
T0	*
_output_shapes
:�
<sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOpAssignVariableOpCsequential_1_bidirectional_1_forward_lstm_1_readvariableop_resource3sequential_1/bidirectional_1/forward_lstm_1/Add:z:0?^sequential_1/bidirectional_1/forward_lstm_1/Add/ReadVariableOp;^sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(|
6sequential_1/bidirectional_1/forward_lstm_1/FloorMod/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
4sequential_1/bidirectional_1/forward_lstm_1/FloorModFloorMod3sequential_1/bidirectional_1/forward_lstm_1/mul:z:0?sequential_1/bidirectional_1/forward_lstm_1/FloorMod/y:output:0*
T0	*
_output_shapes
:�
0sequential_1/bidirectional_1/forward_lstm_1/CastCast8sequential_1/bidirectional_1/forward_lstm_1/FloorMod:z:0*

DstT0*

SrcT0	*
_output_shapes
:�
3sequential_1/bidirectional_1/forward_lstm_1/Shape_1Shape9sequential_1/bidirectional_1/forward_lstm_1/ones_like:y:0*
T0*
_output_shapes
::���
Csequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
Asequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/MulMul9sequential_1/bidirectional_1/forward_lstm_1/ones_like:y:0Lsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/Const:output:0*
T0*(
_output_shapes
:�����������
Zsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Zsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ssequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter4sequential_1/bidirectional_1/forward_lstm_1/Cast:y:0*
Tseed0* 
_output_shapes
::�
ssequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
osequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2<sequential_1/bidirectional_1/forward_lstm_1/Shape_1:output:0ysequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0}sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0|sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:�����������
Zsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/subSubcsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/max:output:0csequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
Zsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/mulMulxsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2:output:0^sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
Vsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniformAddV2^sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/mul:z:0csequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:�����������
Lsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
Jsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/GreaterEqualGreaterEqualZsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/stateless_random_uniform:z:0Usequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
Esequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
Fsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/SelectV2SelectV2Nsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/GreaterEqual:z:0Esequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/Mul:z:0Nsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
7sequential_1/bidirectional_1/forward_lstm_1/ones_like_1OnesLike:sequential_1/bidirectional_1/forward_lstm_1/zeros:output:0*
T0*(
_output_shapes
:�����������
<sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp_1ReadVariableOpCsequential_1_bidirectional_1_forward_lstm_1_readvariableop_resource=^sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp*
_output_shapes
:*
dtype0	u
3sequential_1/bidirectional_1/forward_lstm_1/mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
1sequential_1/bidirectional_1/forward_lstm_1/mul_1MulDsequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp_1:value:0<sequential_1/bidirectional_1/forward_lstm_1/mul_1/y:output:0*
T0	*
_output_shapes
:�
3sequential_1/bidirectional_1/forward_lstm_1/Const_1Const*
_output_shapes
:*
dtype0	*%
valueB	"               �
@sequential_1/bidirectional_1/forward_lstm_1/Add_1/ReadVariableOpReadVariableOpCsequential_1_bidirectional_1_forward_lstm_1_readvariableop_resource=^sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp*
_output_shapes
:*
dtype0	�
1sequential_1/bidirectional_1/forward_lstm_1/Add_1AddV2Hsequential_1/bidirectional_1/forward_lstm_1/Add_1/ReadVariableOp:value:0<sequential_1/bidirectional_1/forward_lstm_1/Const_1:output:0*
T0	*
_output_shapes
:�
>sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp_1AssignVariableOpCsequential_1_bidirectional_1_forward_lstm_1_readvariableop_resource5sequential_1/bidirectional_1/forward_lstm_1/Add_1:z:0A^sequential_1/bidirectional_1/forward_lstm_1/Add_1/ReadVariableOp=^sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp=^sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp_1*
_output_shapes
 *
dtype0	*
validate_shape(~
8sequential_1/bidirectional_1/forward_lstm_1/FloorMod_1/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
6sequential_1/bidirectional_1/forward_lstm_1/FloorMod_1FloorMod5sequential_1/bidirectional_1/forward_lstm_1/mul_1:z:0Asequential_1/bidirectional_1/forward_lstm_1/FloorMod_1/y:output:0*
T0	*
_output_shapes
:�
2sequential_1/bidirectional_1/forward_lstm_1/Cast_1Cast:sequential_1/bidirectional_1/forward_lstm_1/FloorMod_1:z:0*

DstT0*

SrcT0	*
_output_shapes
:�
3sequential_1/bidirectional_1/forward_lstm_1/Shape_2Shape;sequential_1/bidirectional_1/forward_lstm_1/ones_like_1:y:0*
T0*
_output_shapes
::���
Esequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Csequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/MulMul;sequential_1/bidirectional_1/forward_lstm_1/ones_like_1:y:0Nsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/Const:output:0*
T0*(
_output_shapes
:�����������
\sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
\sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
usequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter6sequential_1/bidirectional_1/forward_lstm_1/Cast_1:y:0*
Tseed0* 
_output_shapes
::�
usequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
qsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2<sequential_1/bidirectional_1/forward_lstm_1/Shape_2:output:0{sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0~sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:�����������
\sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/subSubesequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/max:output:0esequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
\sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/mulMulzsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2:output:0`sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
Xsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniformAddV2`sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/mul:z:0esequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:�����������
Nsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
Lsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/GreaterEqualGreaterEqual\sequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/stateless_random_uniform:z:0Wsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
Gsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
Hsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/SelectV2SelectV2Psequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/GreaterEqual:z:0Gsequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/Mul:z:0Psequential_1/bidirectional_1/forward_lstm_1/stateless_dropout_1/Const_1:output:0*
T0*(
_output_shapes
:�����������
:sequential_1/bidirectional_1/forward_lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
5sequential_1/bidirectional_1/forward_lstm_1/transpose	Transpose*sequential_1/embedding_1/GatherV2:output:0Csequential_1/bidirectional_1/forward_lstm_1/transpose/perm:output:0*
T0*,
_output_shapes
:x�����������
:sequential_1/bidirectional_1/forward_lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
6sequential_1/bidirectional_1/forward_lstm_1/ExpandDims
ExpandDimssequential_1/NotEqual_1:z:0Csequential_1/bidirectional_1/forward_lstm_1/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:���������x�
<sequential_1/bidirectional_1/forward_lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
7sequential_1/bidirectional_1/forward_lstm_1/transpose_1	Transpose?sequential_1/bidirectional_1/forward_lstm_1/ExpandDims:output:0Esequential_1/bidirectional_1/forward_lstm_1/transpose_1/perm:output:0*
T0
*+
_output_shapes
:x����������
Gsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Fsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
9sequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2TensorListReservePsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2/element_shape:output:0Osequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
asequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Ssequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor9sequential_1/bidirectional_1/forward_lstm_1/transpose:y:0jsequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Asequential_1/bidirectional_1/forward_lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Csequential_1/bidirectional_1/forward_lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Csequential_1/bidirectional_1/forward_lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential_1/bidirectional_1/forward_lstm_1/strided_slice_2StridedSlice9sequential_1/bidirectional_1/forward_lstm_1/transpose:y:0Jsequential_1/bidirectional_1/forward_lstm_1/strided_slice_2/stack:output:0Lsequential_1/bidirectional_1/forward_lstm_1/strided_slice_2/stack_1:output:0Lsequential_1/bidirectional_1/forward_lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
Ksequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOpTsequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
>sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/MatMulMatMulDsequential_1/bidirectional_1/forward_lstm_1/strided_slice_2:output:0Ssequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Msequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpVsequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_cast_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
@sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/MatMul_1MatMul:sequential_1/bidirectional_1/forward_lstm_1/zeros:output:0Usequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
;sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/addAddV2Hsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/MatMul:product:0Jsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Lsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOpUsequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_1AddV2?sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add:z:0Tsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Gsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
=sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/splitSplitPsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/split/split_dim:output:0Asequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
?sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/SigmoidSigmoidFsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
Asequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Sigmoid_1SigmoidFsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
;sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/mulMulEsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Sigmoid_1:y:0<sequential_1/bidirectional_1/forward_lstm_1/zeros_1:output:0*
T0*(
_output_shapes
:�����������
<sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/TanhTanhFsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/mul_1MulCsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Sigmoid:y:0@sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_2AddV2?sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/mul:z:0Asequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
Asequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Sigmoid_2SigmoidFsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
>sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Tanh_1TanhAsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/mul_2MulEsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Sigmoid_2:y:0Bsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
Isequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Hsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
;sequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_1TensorListReserveRsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_1/element_shape:output:0Qsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���r
0sequential_1/bidirectional_1/forward_lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : x
6sequential_1/bidirectional_1/forward_lstm_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value	B :xr
0sequential_1/bidirectional_1/forward_lstm_1/RankConst*
_output_shapes
: *
dtype0*
value	B : y
7sequential_1/bidirectional_1/forward_lstm_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : y
7sequential_1/bidirectional_1/forward_lstm_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
1sequential_1/bidirectional_1/forward_lstm_1/rangeRange@sequential_1/bidirectional_1/forward_lstm_1/range/start:output:09sequential_1/bidirectional_1/forward_lstm_1/Rank:output:0@sequential_1/bidirectional_1/forward_lstm_1/range/delta:output:0*
_output_shapes
: w
5sequential_1/bidirectional_1/forward_lstm_1/Max/inputConst*
_output_shapes
: *
dtype0*
value	B :x�
/sequential_1/bidirectional_1/forward_lstm_1/MaxMax>sequential_1/bidirectional_1/forward_lstm_1/Max/input:output:0:sequential_1/bidirectional_1/forward_lstm_1/range:output:0*
T0*
_output_shapes
: �
Isequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Hsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
;sequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_2TensorListReserveRsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_2/element_shape:output:0Qsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
csequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Usequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor;sequential_1/bidirectional_1/forward_lstm_1/transpose_1:y:0lsequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
6sequential_1/bidirectional_1/forward_lstm_1/zeros_like	ZerosLikeAsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:�����������
>sequential_1/bidirectional_1/forward_lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
1sequential_1/bidirectional_1/forward_lstm_1/whileWhileGsequential_1/bidirectional_1/forward_lstm_1/while/loop_counter:output:08sequential_1/bidirectional_1/forward_lstm_1/Max:output:09sequential_1/bidirectional_1/forward_lstm_1/time:output:0Dsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2_1:handle:0:sequential_1/bidirectional_1/forward_lstm_1/zeros_like:y:0:sequential_1/bidirectional_1/forward_lstm_1/zeros:output:0<sequential_1/bidirectional_1/forward_lstm_1/zeros_1:output:0csequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0esequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Tsequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_cast_readvariableop_resourceVsequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_cast_1_readvariableop_resourceUsequential_1_bidirectional_1_forward_lstm_1_lstm_cell_1_add_1_readvariableop_resource:sequential_1/bidirectional_1/forward_lstm_1/zeros_like:y:0*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*v
_output_shapesd
b: : : : :����������:����������:����������: : : : : :����������*%
_read_only_resource_inputs
	
*G
body?R=
;sequential_1_bidirectional_1_forward_lstm_1_while_body_1655*G
cond?R=
;sequential_1_bidirectional_1_forward_lstm_1_while_cond_1654*u
output_shapesd
b: : : : :����������:����������:����������: : : : : :����������*
parallel_iterations �
\sequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Nsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack:sequential_1/bidirectional_1/forward_lstm_1/while:output:3esequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x����������*
element_dtype0*
num_elementsx�
Asequential_1/bidirectional_1/forward_lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Csequential_1/bidirectional_1/forward_lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Csequential_1/bidirectional_1/forward_lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential_1/bidirectional_1/forward_lstm_1/strided_slice_3StridedSliceWsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0Jsequential_1/bidirectional_1/forward_lstm_1/strided_slice_3/stack:output:0Lsequential_1/bidirectional_1/forward_lstm_1/strided_slice_3/stack_1:output:0Lsequential_1/bidirectional_1/forward_lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
<sequential_1/bidirectional_1/forward_lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
7sequential_1/bidirectional_1/forward_lstm_1/transpose_2	TransposeWsequential_1/bidirectional_1/forward_lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0Esequential_1/bidirectional_1/forward_lstm_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:���������x��
2sequential_1/bidirectional_1/backward_lstm_1/ShapeShape*sequential_1/embedding_1/GatherV2:output:0*
T0*
_output_shapes
::���
@sequential_1/bidirectional_1/backward_lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Bsequential_1/bidirectional_1/backward_lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Bsequential_1/bidirectional_1/backward_lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
:sequential_1/bidirectional_1/backward_lstm_1/strided_sliceStridedSlice;sequential_1/bidirectional_1/backward_lstm_1/Shape:output:0Isequential_1/bidirectional_1/backward_lstm_1/strided_slice/stack:output:0Ksequential_1/bidirectional_1/backward_lstm_1/strided_slice/stack_1:output:0Ksequential_1/bidirectional_1/backward_lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
;sequential_1/bidirectional_1/backward_lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
9sequential_1/bidirectional_1/backward_lstm_1/zeros/packedPackCsequential_1/bidirectional_1/backward_lstm_1/strided_slice:output:0Dsequential_1/bidirectional_1/backward_lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:}
8sequential_1/bidirectional_1/backward_lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
2sequential_1/bidirectional_1/backward_lstm_1/zerosFillBsequential_1/bidirectional_1/backward_lstm_1/zeros/packed:output:0Asequential_1/bidirectional_1/backward_lstm_1/zeros/Const:output:0*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/backward_lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
;sequential_1/bidirectional_1/backward_lstm_1/zeros_1/packedPackCsequential_1/bidirectional_1/backward_lstm_1/strided_slice:output:0Fsequential_1/bidirectional_1/backward_lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
:sequential_1/bidirectional_1/backward_lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
4sequential_1/bidirectional_1/backward_lstm_1/zeros_1FillDsequential_1/bidirectional_1/backward_lstm_1/zeros_1/packed:output:0Csequential_1/bidirectional_1/backward_lstm_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:�����������
Bsequential_1/bidirectional_1/backward_lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
Dsequential_1/bidirectional_1/backward_lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
Dsequential_1/bidirectional_1/backward_lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
<sequential_1/bidirectional_1/backward_lstm_1/strided_slice_1StridedSlice*sequential_1/embedding_1/GatherV2:output:0Ksequential_1/bidirectional_1/backward_lstm_1/strided_slice_1/stack:output:0Msequential_1/bidirectional_1/backward_lstm_1/strided_slice_1/stack_1:output:0Msequential_1/bidirectional_1/backward_lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
6sequential_1/bidirectional_1/backward_lstm_1/ones_likeOnesLikeEsequential_1/bidirectional_1/backward_lstm_1/strided_slice_1:output:0*
T0*(
_output_shapes
:�����������
;sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOpReadVariableOpDsequential_1_bidirectional_1_backward_lstm_1_readvariableop_resource*
_output_shapes
:*
dtype0	t
2sequential_1/bidirectional_1/backward_lstm_1/mul/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
0sequential_1/bidirectional_1/backward_lstm_1/mulMulCsequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp:value:0;sequential_1/bidirectional_1/backward_lstm_1/mul/y:output:0*
T0	*
_output_shapes
:�
2sequential_1/bidirectional_1/backward_lstm_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"               �
?sequential_1/bidirectional_1/backward_lstm_1/Add/ReadVariableOpReadVariableOpDsequential_1_bidirectional_1_backward_lstm_1_readvariableop_resource*
_output_shapes
:*
dtype0	�
0sequential_1/bidirectional_1/backward_lstm_1/AddAddV2Gsequential_1/bidirectional_1/backward_lstm_1/Add/ReadVariableOp:value:0;sequential_1/bidirectional_1/backward_lstm_1/Const:output:0*
T0	*
_output_shapes
:�
=sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOpAssignVariableOpDsequential_1_bidirectional_1_backward_lstm_1_readvariableop_resource4sequential_1/bidirectional_1/backward_lstm_1/Add:z:0@^sequential_1/bidirectional_1/backward_lstm_1/Add/ReadVariableOp<^sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(}
7sequential_1/bidirectional_1/backward_lstm_1/FloorMod/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
5sequential_1/bidirectional_1/backward_lstm_1/FloorModFloorMod4sequential_1/bidirectional_1/backward_lstm_1/mul:z:0@sequential_1/bidirectional_1/backward_lstm_1/FloorMod/y:output:0*
T0	*
_output_shapes
:�
1sequential_1/bidirectional_1/backward_lstm_1/CastCast9sequential_1/bidirectional_1/backward_lstm_1/FloorMod:z:0*

DstT0*

SrcT0	*
_output_shapes
:�
4sequential_1/bidirectional_1/backward_lstm_1/Shape_1Shape:sequential_1/bidirectional_1/backward_lstm_1/ones_like:y:0*
T0*
_output_shapes
::���
Dsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
Bsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/MulMul:sequential_1/bidirectional_1/backward_lstm_1/ones_like:y:0Msequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/Const:output:0*
T0*(
_output_shapes
:�����������
[sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
[sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
tsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter5sequential_1/bidirectional_1/backward_lstm_1/Cast:y:0*
Tseed0* 
_output_shapes
::�
tsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
psequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2=sequential_1/bidirectional_1/backward_lstm_1/Shape_1:output:0zsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0~sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0}sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:�����������
[sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/subSubdsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/max:output:0dsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
[sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/mulMulysequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2:output:0_sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
Wsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniformAddV2_sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/mul:z:0dsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:�����������
Msequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
Ksequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/GreaterEqualGreaterEqual[sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/stateless_random_uniform:z:0Vsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
Fsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
Gsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/SelectV2SelectV2Osequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/GreaterEqual:z:0Fsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/Mul:z:0Osequential_1/bidirectional_1/backward_lstm_1/stateless_dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
8sequential_1/bidirectional_1/backward_lstm_1/ones_like_1OnesLike;sequential_1/bidirectional_1/backward_lstm_1/zeros:output:0*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp_1ReadVariableOpDsequential_1_bidirectional_1_backward_lstm_1_readvariableop_resource>^sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp*
_output_shapes
:*
dtype0	v
4sequential_1/bidirectional_1/backward_lstm_1/mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
2sequential_1/bidirectional_1/backward_lstm_1/mul_1MulEsequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp_1:value:0=sequential_1/bidirectional_1/backward_lstm_1/mul_1/y:output:0*
T0	*
_output_shapes
:�
4sequential_1/bidirectional_1/backward_lstm_1/Const_1Const*
_output_shapes
:*
dtype0	*%
valueB	"               �
Asequential_1/bidirectional_1/backward_lstm_1/Add_1/ReadVariableOpReadVariableOpDsequential_1_bidirectional_1_backward_lstm_1_readvariableop_resource>^sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp*
_output_shapes
:*
dtype0	�
2sequential_1/bidirectional_1/backward_lstm_1/Add_1AddV2Isequential_1/bidirectional_1/backward_lstm_1/Add_1/ReadVariableOp:value:0=sequential_1/bidirectional_1/backward_lstm_1/Const_1:output:0*
T0	*
_output_shapes
:�
?sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp_1AssignVariableOpDsequential_1_bidirectional_1_backward_lstm_1_readvariableop_resource6sequential_1/bidirectional_1/backward_lstm_1/Add_1:z:0B^sequential_1/bidirectional_1/backward_lstm_1/Add_1/ReadVariableOp>^sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp>^sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp_1*
_output_shapes
 *
dtype0	*
validate_shape(
9sequential_1/bidirectional_1/backward_lstm_1/FloorMod_1/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
7sequential_1/bidirectional_1/backward_lstm_1/FloorMod_1FloorMod6sequential_1/bidirectional_1/backward_lstm_1/mul_1:z:0Bsequential_1/bidirectional_1/backward_lstm_1/FloorMod_1/y:output:0*
T0	*
_output_shapes
:�
3sequential_1/bidirectional_1/backward_lstm_1/Cast_1Cast;sequential_1/bidirectional_1/backward_lstm_1/FloorMod_1:z:0*

DstT0*

SrcT0	*
_output_shapes
:�
4sequential_1/bidirectional_1/backward_lstm_1/Shape_2Shape<sequential_1/bidirectional_1/backward_lstm_1/ones_like_1:y:0*
T0*
_output_shapes
::���
Fsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/MulMul<sequential_1/bidirectional_1/backward_lstm_1/ones_like_1:y:0Osequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/Const:output:0*
T0*(
_output_shapes
:�����������
]sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
]sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
vsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter7sequential_1/bidirectional_1/backward_lstm_1/Cast_1:y:0*
Tseed0* 
_output_shapes
::�
vsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
rsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2=sequential_1/bidirectional_1/backward_lstm_1/Shape_2:output:0|sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0�sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:�����������
]sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/subSubfsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/max:output:0fsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
]sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/mulMul{sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2:output:0asequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
Ysequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniformAddV2asequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/mul:z:0fsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:�����������
Osequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
Msequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/GreaterEqualGreaterEqual]sequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/stateless_random_uniform:z:0Xsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
Hsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
Isequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/SelectV2SelectV2Qsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/GreaterEqual:z:0Hsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/Mul:z:0Qsequential_1/bidirectional_1/backward_lstm_1/stateless_dropout_1/Const_1:output:0*
T0*(
_output_shapes
:�����������
;sequential_1/bidirectional_1/backward_lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
6sequential_1/bidirectional_1/backward_lstm_1/transpose	Transpose*sequential_1/embedding_1/GatherV2:output:0Dsequential_1/bidirectional_1/backward_lstm_1/transpose/perm:output:0*
T0*,
_output_shapes
:x�����������
;sequential_1/bidirectional_1/backward_lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
7sequential_1/bidirectional_1/backward_lstm_1/ExpandDims
ExpandDimssequential_1/NotEqual_1:z:0Dsequential_1/bidirectional_1/backward_lstm_1/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:���������x�
=sequential_1/bidirectional_1/backward_lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
8sequential_1/bidirectional_1/backward_lstm_1/transpose_1	Transpose@sequential_1/bidirectional_1/backward_lstm_1/ExpandDims:output:0Fsequential_1/bidirectional_1/backward_lstm_1/transpose_1/perm:output:0*
T0
*+
_output_shapes
:x����������
Hsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Gsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
:sequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2TensorListReserveQsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2/element_shape:output:0Psequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;sequential_1/bidirectional_1/backward_lstm_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
6sequential_1/bidirectional_1/backward_lstm_1/ReverseV2	ReverseV2:sequential_1/bidirectional_1/backward_lstm_1/transpose:y:0Dsequential_1/bidirectional_1/backward_lstm_1/ReverseV2/axis:output:0*
T0*,
_output_shapes
:x�����������
bsequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Tsequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor?sequential_1/bidirectional_1/backward_lstm_1/ReverseV2:output:0ksequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsequential_1/bidirectional_1/backward_lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Dsequential_1/bidirectional_1/backward_lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Dsequential_1/bidirectional_1/backward_lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
<sequential_1/bidirectional_1/backward_lstm_1/strided_slice_2StridedSlice:sequential_1/bidirectional_1/backward_lstm_1/transpose:y:0Ksequential_1/bidirectional_1/backward_lstm_1/strided_slice_2/stack:output:0Msequential_1/bidirectional_1/backward_lstm_1/strided_slice_2/stack_1:output:0Msequential_1/bidirectional_1/backward_lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
Lsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOpUsequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
?sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/MatMulMatMulEsequential_1/bidirectional_1/backward_lstm_1/strided_slice_2:output:0Tsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Nsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpWsequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_cast_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Asequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/MatMul_1MatMul;sequential_1/bidirectional_1/backward_lstm_1/zeros:output:0Vsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/addAddV2Isequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/MatMul:product:0Ksequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Msequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOpVsequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
>sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_1AddV2@sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add:z:0Usequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Hsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
>sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/splitSplitQsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/split/split_dim:output:0Bsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
@sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/SigmoidSigmoidGsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
Bsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Sigmoid_1SigmoidGsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
<sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/mulMulFsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Sigmoid_1:y:0=sequential_1/bidirectional_1/backward_lstm_1/zeros_1:output:0*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/TanhTanhGsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
>sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/mul_1MulDsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Sigmoid:y:0Asequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
>sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_2AddV2@sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/mul:z:0Bsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
Bsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Sigmoid_2SigmoidGsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
?sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Tanh_1TanhBsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
>sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/mul_2MulFsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Sigmoid_2:y:0Csequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
Jsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Isequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
<sequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_1TensorListReserveSsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_1/element_shape:output:0Rsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���s
1sequential_1/bidirectional_1/backward_lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : y
7sequential_1/bidirectional_1/backward_lstm_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value	B :xs
1sequential_1/bidirectional_1/backward_lstm_1/RankConst*
_output_shapes
: *
dtype0*
value	B : z
8sequential_1/bidirectional_1/backward_lstm_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : z
8sequential_1/bidirectional_1/backward_lstm_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
2sequential_1/bidirectional_1/backward_lstm_1/rangeRangeAsequential_1/bidirectional_1/backward_lstm_1/range/start:output:0:sequential_1/bidirectional_1/backward_lstm_1/Rank:output:0Asequential_1/bidirectional_1/backward_lstm_1/range/delta:output:0*
_output_shapes
: x
6sequential_1/bidirectional_1/backward_lstm_1/Max/inputConst*
_output_shapes
: *
dtype0*
value	B :x�
0sequential_1/bidirectional_1/backward_lstm_1/MaxMax?sequential_1/bidirectional_1/backward_lstm_1/Max/input:output:0;sequential_1/bidirectional_1/backward_lstm_1/range:output:0*
T0*
_output_shapes
: �
=sequential_1/bidirectional_1/backward_lstm_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: �
8sequential_1/bidirectional_1/backward_lstm_1/ReverseV2_1	ReverseV2<sequential_1/bidirectional_1/backward_lstm_1/transpose_1:y:0Fsequential_1/bidirectional_1/backward_lstm_1/ReverseV2_1/axis:output:0*
T0
*+
_output_shapes
:x����������
Jsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Isequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
<sequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_2TensorListReserveSsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_2/element_shape:output:0Rsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
dsequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Vsequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorAsequential_1/bidirectional_1/backward_lstm_1/ReverseV2_1:output:0msequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
7sequential_1/bidirectional_1/backward_lstm_1/zeros_like	ZerosLikeBsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:�����������
?sequential_1/bidirectional_1/backward_lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
2sequential_1/bidirectional_1/backward_lstm_1/whileWhileHsequential_1/bidirectional_1/backward_lstm_1/while/loop_counter:output:09sequential_1/bidirectional_1/backward_lstm_1/Max:output:0:sequential_1/bidirectional_1/backward_lstm_1/time:output:0Esequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2_1:handle:0;sequential_1/bidirectional_1/backward_lstm_1/zeros_like:y:0;sequential_1/bidirectional_1/backward_lstm_1/zeros:output:0=sequential_1/bidirectional_1/backward_lstm_1/zeros_1:output:0dsequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0fsequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Usequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_cast_readvariableop_resourceWsequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_cast_1_readvariableop_resourceVsequential_1_bidirectional_1_backward_lstm_1_lstm_cell_1_add_1_readvariableop_resource;sequential_1/bidirectional_1/backward_lstm_1/zeros_like:y:0*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*v
_output_shapesd
b: : : : :����������:����������:����������: : : : : :����������*%
_read_only_resource_inputs
	
*H
body@R>
<sequential_1_bidirectional_1_backward_lstm_1_while_body_1888*H
cond@R>
<sequential_1_bidirectional_1_backward_lstm_1_while_cond_1887*u
output_shapesd
b: : : : :����������:����������:����������: : : : : :����������*
parallel_iterations �
]sequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Osequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack;sequential_1/bidirectional_1/backward_lstm_1/while:output:3fsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:x����������*
element_dtype0*
num_elementsx�
Bsequential_1/bidirectional_1/backward_lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Dsequential_1/bidirectional_1/backward_lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dsequential_1/bidirectional_1/backward_lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
<sequential_1/bidirectional_1/backward_lstm_1/strided_slice_3StridedSliceXsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0Ksequential_1/bidirectional_1/backward_lstm_1/strided_slice_3/stack:output:0Msequential_1/bidirectional_1/backward_lstm_1/strided_slice_3/stack_1:output:0Msequential_1/bidirectional_1/backward_lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
=sequential_1/bidirectional_1/backward_lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
8sequential_1/bidirectional_1/backward_lstm_1/transpose_2	TransposeXsequential_1/bidirectional_1/backward_lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0Fsequential_1/bidirectional_1/backward_lstm_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:���������x�u
+sequential_1/bidirectional_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
&sequential_1/bidirectional_1/ReverseV2	ReverseV2<sequential_1/bidirectional_1/backward_lstm_1/transpose_2:y:04sequential_1/bidirectional_1/ReverseV2/axis:output:0*
T0*,
_output_shapes
:���������x�s
(sequential_1/bidirectional_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
#sequential_1/bidirectional_1/concatConcatV2;sequential_1/bidirectional_1/forward_lstm_1/transpose_2:y:0/sequential_1/bidirectional_1/ReverseV2:output:01sequential_1/bidirectional_1/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������x��
6sequential_1/batch_normalization_1/Cast/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8sequential_1/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8sequential_1/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8sequential_1/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0w
2sequential_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0sequential_1/batch_normalization_1/batchnorm/addAddV2@sequential_1/batch_normalization_1/Cast_1/ReadVariableOp:value:0;sequential_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2sequential_1/batch_normalization_1/batchnorm/RsqrtRsqrt4sequential_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0sequential_1/batch_normalization_1/batchnorm/mulMul6sequential_1/batch_normalization_1/batchnorm/Rsqrt:y:0@sequential_1/batch_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2sequential_1/batch_normalization_1/batchnorm/mul_1Mul,sequential_1/bidirectional_1/concat:output:04sequential_1/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������x��
2sequential_1/batch_normalization_1/batchnorm/mul_2Mul>sequential_1/batch_normalization_1/Cast/ReadVariableOp:value:04sequential_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0sequential_1/batch_normalization_1/batchnorm/subSub@sequential_1/batch_normalization_1/Cast_3/ReadVariableOp:value:06sequential_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2sequential_1/batch_normalization_1/batchnorm/add_1AddV26sequential_1/batch_normalization_1/batchnorm/mul_1:z:04sequential_1/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������x��
5sequential_1/bidirectional_1_2/forward_lstm_1_1/ShapeShape6sequential_1/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
Csequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Esequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Esequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=sequential_1/bidirectional_1_2/forward_lstm_1_1/strided_sliceStridedSlice>sequential_1/bidirectional_1_2/forward_lstm_1_1/Shape:output:0Lsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice/stack:output:0Nsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice/stack_1:output:0Nsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
>sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
<sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros/packedPackFsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice:output:0Gsequential_1/bidirectional_1_2/forward_lstm_1_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:�
;sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
5sequential_1/bidirectional_1_2/forward_lstm_1_1/zerosFillEsequential_1/bidirectional_1_2/forward_lstm_1_1/zeros/packed:output:0Dsequential_1/bidirectional_1_2/forward_lstm_1_1/zeros/Const:output:0*
T0*'
_output_shapes
:���������@�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
>sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1/packedPackFsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice:output:0Isequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:�
=sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
7sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1FillGsequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1/packed:output:0Fsequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@�
Esequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
?sequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_1StridedSlice6sequential_1/batch_normalization_1/batchnorm/add_1:z:0Nsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_1/stack:output:0Psequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_1/stack_1:output:0Psequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
9sequential_1/bidirectional_1_2/forward_lstm_1_1/ones_likeOnesLikeHsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_1:output:0*
T0*(
_output_shapes
:�����������
>sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOpReadVariableOpGsequential_1_bidirectional_1_2_forward_lstm_1_1_readvariableop_resource*
_output_shapes
:*
dtype0	w
5sequential_1/bidirectional_1_2/forward_lstm_1_1/mul/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
3sequential_1/bidirectional_1_2/forward_lstm_1_1/mulMulFsequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp:value:0>sequential_1/bidirectional_1_2/forward_lstm_1_1/mul/y:output:0*
T0	*
_output_shapes
:�
5sequential_1/bidirectional_1_2/forward_lstm_1_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"               �
Bsequential_1/bidirectional_1_2/forward_lstm_1_1/Add/ReadVariableOpReadVariableOpGsequential_1_bidirectional_1_2_forward_lstm_1_1_readvariableop_resource*
_output_shapes
:*
dtype0	�
3sequential_1/bidirectional_1_2/forward_lstm_1_1/AddAddV2Jsequential_1/bidirectional_1_2/forward_lstm_1_1/Add/ReadVariableOp:value:0>sequential_1/bidirectional_1_2/forward_lstm_1_1/Const:output:0*
T0	*
_output_shapes
:�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOpAssignVariableOpGsequential_1_bidirectional_1_2_forward_lstm_1_1_readvariableop_resource7sequential_1/bidirectional_1_2/forward_lstm_1_1/Add:z:0C^sequential_1/bidirectional_1_2/forward_lstm_1_1/Add/ReadVariableOp?^sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(�
:sequential_1/bidirectional_1_2/forward_lstm_1_1/FloorMod/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
8sequential_1/bidirectional_1_2/forward_lstm_1_1/FloorModFloorMod7sequential_1/bidirectional_1_2/forward_lstm_1_1/mul:z:0Csequential_1/bidirectional_1_2/forward_lstm_1_1/FloorMod/y:output:0*
T0	*
_output_shapes
:�
4sequential_1/bidirectional_1_2/forward_lstm_1_1/CastCast<sequential_1/bidirectional_1_2/forward_lstm_1_1/FloorMod:z:0*

DstT0*

SrcT0	*
_output_shapes
:�
7sequential_1/bidirectional_1_2/forward_lstm_1_1/Shape_1Shape=sequential_1/bidirectional_1_2/forward_lstm_1_1/ones_like:y:0*
T0*
_output_shapes
::���
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
Esequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/MulMul=sequential_1/bidirectional_1_2/forward_lstm_1_1/ones_like:y:0Psequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/Const:output:0*
T0*(
_output_shapes
:�����������
^sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
^sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
wsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter8sequential_1/bidirectional_1_2/forward_lstm_1_1/Cast:y:0*
Tseed0* 
_output_shapes
::�
wsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
ssequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2@sequential_1/bidirectional_1_2/forward_lstm_1_1/Shape_1:output:0}sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0�sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0�sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:�����������
^sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/subSubgsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/max:output:0gsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
^sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/mulMul|sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2:output:0bsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
Zsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniformAddV2bsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/mul:z:0gsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:�����������
Psequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
Nsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/GreaterEqualGreaterEqual^sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/stateless_random_uniform:z:0Ysequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
Isequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
Jsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/SelectV2SelectV2Rsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/GreaterEqual:z:0Isequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/Mul:z:0Rsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
;sequential_1/bidirectional_1_2/forward_lstm_1_1/ones_like_1OnesLike>sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros:output:0*
T0*'
_output_shapes
:���������@�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp_1ReadVariableOpGsequential_1_bidirectional_1_2_forward_lstm_1_1_readvariableop_resourceA^sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp*
_output_shapes
:*
dtype0	y
7sequential_1/bidirectional_1_2/forward_lstm_1_1/mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
5sequential_1/bidirectional_1_2/forward_lstm_1_1/mul_1MulHsequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp_1:value:0@sequential_1/bidirectional_1_2/forward_lstm_1_1/mul_1/y:output:0*
T0	*
_output_shapes
:�
7sequential_1/bidirectional_1_2/forward_lstm_1_1/Const_1Const*
_output_shapes
:*
dtype0	*%
valueB	"               �
Dsequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1/ReadVariableOpReadVariableOpGsequential_1_bidirectional_1_2_forward_lstm_1_1_readvariableop_resourceA^sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp*
_output_shapes
:*
dtype0	�
5sequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1AddV2Lsequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1/ReadVariableOp:value:0@sequential_1/bidirectional_1_2/forward_lstm_1_1/Const_1:output:0*
T0	*
_output_shapes
:�
Bsequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp_1AssignVariableOpGsequential_1_bidirectional_1_2_forward_lstm_1_1_readvariableop_resource9sequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1:z:0E^sequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1/ReadVariableOpA^sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOpA^sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp_1*
_output_shapes
 *
dtype0	*
validate_shape(�
<sequential_1/bidirectional_1_2/forward_lstm_1_1/FloorMod_1/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
:sequential_1/bidirectional_1_2/forward_lstm_1_1/FloorMod_1FloorMod9sequential_1/bidirectional_1_2/forward_lstm_1_1/mul_1:z:0Esequential_1/bidirectional_1_2/forward_lstm_1_1/FloorMod_1/y:output:0*
T0	*
_output_shapes
:�
6sequential_1/bidirectional_1_2/forward_lstm_1_1/Cast_1Cast>sequential_1/bidirectional_1_2/forward_lstm_1_1/FloorMod_1:z:0*

DstT0*

SrcT0	*
_output_shapes
:�
7sequential_1/bidirectional_1_2/forward_lstm_1_1/Shape_2Shape?sequential_1/bidirectional_1_2/forward_lstm_1_1/ones_like_1:y:0*
T0*
_output_shapes
::���
Isequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/MulMul?sequential_1/bidirectional_1_2/forward_lstm_1_1/ones_like_1:y:0Rsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/Const:output:0*
T0*'
_output_shapes
:���������@�
`sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
`sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ysequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter:sequential_1/bidirectional_1_2/forward_lstm_1_1/Cast_1:y:0*
Tseed0* 
_output_shapes
::�
ysequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
usequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2@sequential_1/bidirectional_1_2/forward_lstm_1_1/Shape_2:output:0sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0�sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0�sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:���������@�
`sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/subSubisequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/max:output:0isequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
`sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/mulMul~sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2:output:0dsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/sub:z:0*
T0*'
_output_shapes
:���������@�
\sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniformAddV2dsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/mul:z:0isequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/min:output:0*
T0*'
_output_shapes
:���������@�
Rsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
Psequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/GreaterEqualGreaterEqual`sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/stateless_random_uniform:z:0[sequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
Ksequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
Lsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/SelectV2SelectV2Tsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/GreaterEqual:z:0Ksequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/Mul:z:0Tsequential_1/bidirectional_1_2/forward_lstm_1_1/stateless_dropout_1/Const_1:output:0*
T0*'
_output_shapes
:���������@�
>sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
9sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose	Transpose6sequential_1/batch_normalization_1/batchnorm/add_1:z:0Gsequential_1/bidirectional_1_2/forward_lstm_1_1/transpose/perm:output:0*
T0*,
_output_shapes
:x�����������
>sequential_1/bidirectional_1_2/forward_lstm_1_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
:sequential_1/bidirectional_1_2/forward_lstm_1_1/ExpandDims
ExpandDimssequential_1/NotEqual_1:z:0Gsequential_1/bidirectional_1_2/forward_lstm_1_1/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:���������x�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
;sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose_1	TransposeCsequential_1/bidirectional_1_2/forward_lstm_1_1/ExpandDims:output:0Isequential_1/bidirectional_1_2/forward_lstm_1_1/transpose_1/perm:output:0*
T0
*+
_output_shapes
:x����������
Ksequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Jsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
=sequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2TensorListReserveTsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2/element_shape:output:0Ssequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
esequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Wsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor=sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose:y:0nsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Esequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?sequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_2StridedSlice=sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose:y:0Nsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_2/stack:output:0Psequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_2/stack_1:output:0Psequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
Osequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOpXsequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Bsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/MatMulMatMulHsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_2:output:0Wsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Qsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpZsequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_cast_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
Dsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/MatMul_1MatMul>sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros:output:0Ysequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
?sequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/addAddV2Lsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/MatMul:product:0Nsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Psequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOpYsequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Asequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_1AddV2Csequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add:z:0Xsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Ksequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Asequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/splitSplitTsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/split/split_dim:output:0Esequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split�
Csequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/SigmoidSigmoidJsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@�
Esequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Sigmoid_1SigmoidJsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@�
?sequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/mulMulIsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Sigmoid_1:y:0@sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1:output:0*
T0*'
_output_shapes
:���������@�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/TanhTanhJsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/mul_1MulGsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Sigmoid:y:0Dsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_2AddV2Csequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/mul:z:0Esequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@�
Esequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Sigmoid_2SigmoidJsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@�
Bsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Tanh_1TanhEsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/mul_2MulIsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Sigmoid_2:y:0Fsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
Msequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
Lsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
?sequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_1TensorListReserveVsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_1/element_shape:output:0Usequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
4sequential_1/bidirectional_1_2/forward_lstm_1_1/timeConst*
_output_shapes
: *
dtype0*
value	B : |
:sequential_1/bidirectional_1_2/forward_lstm_1_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value	B :xv
4sequential_1/bidirectional_1_2/forward_lstm_1_1/RankConst*
_output_shapes
: *
dtype0*
value	B : }
;sequential_1/bidirectional_1_2/forward_lstm_1_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : }
;sequential_1/bidirectional_1_2/forward_lstm_1_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
5sequential_1/bidirectional_1_2/forward_lstm_1_1/rangeRangeDsequential_1/bidirectional_1_2/forward_lstm_1_1/range/start:output:0=sequential_1/bidirectional_1_2/forward_lstm_1_1/Rank:output:0Dsequential_1/bidirectional_1_2/forward_lstm_1_1/range/delta:output:0*
_output_shapes
: {
9sequential_1/bidirectional_1_2/forward_lstm_1_1/Max/inputConst*
_output_shapes
: *
dtype0*
value	B :x�
3sequential_1/bidirectional_1_2/forward_lstm_1_1/MaxMaxBsequential_1/bidirectional_1_2/forward_lstm_1_1/Max/input:output:0>sequential_1/bidirectional_1_2/forward_lstm_1_1/range:output:0*
T0*
_output_shapes
: �
Msequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Lsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
?sequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_2TensorListReserveVsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_2/element_shape:output:0Usequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
gsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Ysequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor?sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose_1:y:0psequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
:sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_like	ZerosLikeEsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/mul_2:z:0*
T0*'
_output_shapes
:���������@�
Bsequential_1/bidirectional_1_2/forward_lstm_1_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
5sequential_1/bidirectional_1_2/forward_lstm_1_1/whileWhileKsequential_1/bidirectional_1_2/forward_lstm_1_1/while/loop_counter:output:0<sequential_1/bidirectional_1_2/forward_lstm_1_1/Max:output:0=sequential_1/bidirectional_1_2/forward_lstm_1_1/time:output:0Hsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2_1:handle:0>sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_like:y:0>sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros:output:0@sequential_1/bidirectional_1_2/forward_lstm_1_1/zeros_1:output:0gsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0isequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Xsequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_cast_readvariableop_resourceZsequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_cast_1_readvariableop_resourceYsequential_1_bidirectional_1_2_forward_lstm_1_1_lstm_cell_1_add_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*_
_output_shapesM
K: : : : :���������@:���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*K
bodyCRA
?sequential_1_bidirectional_1_2_forward_lstm_1_1_while_body_2137*K
condCRA
?sequential_1_bidirectional_1_2_forward_lstm_1_1_while_cond_2136*^
output_shapesM
K: : : : :���������@:���������@:���������@: : : : : *
parallel_iterations �
`sequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
Rsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2Stack/TensorListStackTensorListStack>sequential_1/bidirectional_1_2/forward_lstm_1_1/while:output:3isequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elements�
Esequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?sequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_3StridedSlice[sequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_3/stack:output:0Psequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_3/stack_1:output:0Psequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
;sequential_1/bidirectional_1_2/forward_lstm_1_1/transpose_2	Transpose[sequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayV2Stack/TensorListStack:tensor:0Isequential_1/bidirectional_1_2/forward_lstm_1_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:���������@�
6sequential_1/bidirectional_1_2/backward_lstm_1_1/ShapeShape6sequential_1/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
Dsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
>sequential_1/bidirectional_1_2/backward_lstm_1_1/strided_sliceStridedSlice?sequential_1/bidirectional_1_2/backward_lstm_1_1/Shape:output:0Msequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice/stack:output:0Osequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice/stack_1:output:0Osequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
?sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
=sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros/packedPackGsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice:output:0Hsequential_1/bidirectional_1_2/backward_lstm_1_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:�
<sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
6sequential_1/bidirectional_1_2/backward_lstm_1_1/zerosFillFsequential_1/bidirectional_1_2/backward_lstm_1_1/zeros/packed:output:0Esequential_1/bidirectional_1_2/backward_lstm_1_1/zeros/Const:output:0*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
?sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1/packedPackGsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice:output:0Jsequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:�
>sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
8sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1FillHsequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1/packed:output:0Gsequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@�
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
@sequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_1StridedSlice6sequential_1/batch_normalization_1/batchnorm/add_1:z:0Osequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_1/stack:output:0Qsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_1/stack_1:output:0Qsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
:sequential_1/bidirectional_1_2/backward_lstm_1_1/ones_likeOnesLikeIsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_1:output:0*
T0*(
_output_shapes
:�����������
?sequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOpReadVariableOpHsequential_1_bidirectional_1_2_backward_lstm_1_1_readvariableop_resource*
_output_shapes
:*
dtype0	x
6sequential_1/bidirectional_1_2/backward_lstm_1_1/mul/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
4sequential_1/bidirectional_1_2/backward_lstm_1_1/mulMulGsequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp:value:0?sequential_1/bidirectional_1_2/backward_lstm_1_1/mul/y:output:0*
T0	*
_output_shapes
:�
6sequential_1/bidirectional_1_2/backward_lstm_1_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"               �
Csequential_1/bidirectional_1_2/backward_lstm_1_1/Add/ReadVariableOpReadVariableOpHsequential_1_bidirectional_1_2_backward_lstm_1_1_readvariableop_resource*
_output_shapes
:*
dtype0	�
4sequential_1/bidirectional_1_2/backward_lstm_1_1/AddAddV2Ksequential_1/bidirectional_1_2/backward_lstm_1_1/Add/ReadVariableOp:value:0?sequential_1/bidirectional_1_2/backward_lstm_1_1/Const:output:0*
T0	*
_output_shapes
:�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOpAssignVariableOpHsequential_1_bidirectional_1_2_backward_lstm_1_1_readvariableop_resource8sequential_1/bidirectional_1_2/backward_lstm_1_1/Add:z:0D^sequential_1/bidirectional_1_2/backward_lstm_1_1/Add/ReadVariableOp@^sequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(�
;sequential_1/bidirectional_1_2/backward_lstm_1_1/FloorMod/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
9sequential_1/bidirectional_1_2/backward_lstm_1_1/FloorModFloorMod8sequential_1/bidirectional_1_2/backward_lstm_1_1/mul:z:0Dsequential_1/bidirectional_1_2/backward_lstm_1_1/FloorMod/y:output:0*
T0	*
_output_shapes
:�
5sequential_1/bidirectional_1_2/backward_lstm_1_1/CastCast=sequential_1/bidirectional_1_2/backward_lstm_1_1/FloorMod:z:0*

DstT0*

SrcT0	*
_output_shapes
:�
8sequential_1/bidirectional_1_2/backward_lstm_1_1/Shape_1Shape>sequential_1/bidirectional_1_2/backward_lstm_1_1/ones_like:y:0*
T0*
_output_shapes
::���
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/MulMul>sequential_1/bidirectional_1_2/backward_lstm_1_1/ones_like:y:0Qsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/Const:output:0*
T0*(
_output_shapes
:�����������
_sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
_sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
xsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter9sequential_1/bidirectional_1_2/backward_lstm_1_1/Cast:y:0*
Tseed0* 
_output_shapes
::�
xsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
tsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Asequential_1/bidirectional_1_2/backward_lstm_1_1/Shape_1:output:0~sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0�sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0�sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:�����������
_sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/subSubhsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/max:output:0hsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
_sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/mulMul}sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/StatelessRandomUniformV2:output:0csequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
[sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniformAddV2csequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/mul:z:0hsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:�����������
Qsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
Osequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/GreaterEqualGreaterEqual_sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/stateless_random_uniform:z:0Zsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
Jsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
Ksequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/SelectV2SelectV2Ssequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/GreaterEqual:z:0Jsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/Mul:z:0Ssequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
<sequential_1/bidirectional_1_2/backward_lstm_1_1/ones_like_1OnesLike?sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros:output:0*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp_1ReadVariableOpHsequential_1_bidirectional_1_2_backward_lstm_1_1_readvariableop_resourceB^sequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp*
_output_shapes
:*
dtype0	z
8sequential_1/bidirectional_1_2/backward_lstm_1_1/mul_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
6sequential_1/bidirectional_1_2/backward_lstm_1_1/mul_1MulIsequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp_1:value:0Asequential_1/bidirectional_1_2/backward_lstm_1_1/mul_1/y:output:0*
T0	*
_output_shapes
:�
8sequential_1/bidirectional_1_2/backward_lstm_1_1/Const_1Const*
_output_shapes
:*
dtype0	*%
valueB	"               �
Esequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1/ReadVariableOpReadVariableOpHsequential_1_bidirectional_1_2_backward_lstm_1_1_readvariableop_resourceB^sequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp*
_output_shapes
:*
dtype0	�
6sequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1AddV2Msequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1/ReadVariableOp:value:0Asequential_1/bidirectional_1_2/backward_lstm_1_1/Const_1:output:0*
T0	*
_output_shapes
:�
Csequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp_1AssignVariableOpHsequential_1_bidirectional_1_2_backward_lstm_1_1_readvariableop_resource:sequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1:z:0F^sequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1/ReadVariableOpB^sequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOpB^sequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp_1*
_output_shapes
 *
dtype0	*
validate_shape(�
=sequential_1/bidirectional_1_2/backward_lstm_1_1/FloorMod_1/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
;sequential_1/bidirectional_1_2/backward_lstm_1_1/FloorMod_1FloorMod:sequential_1/bidirectional_1_2/backward_lstm_1_1/mul_1:z:0Fsequential_1/bidirectional_1_2/backward_lstm_1_1/FloorMod_1/y:output:0*
T0	*
_output_shapes
:�
7sequential_1/bidirectional_1_2/backward_lstm_1_1/Cast_1Cast?sequential_1/bidirectional_1_2/backward_lstm_1_1/FloorMod_1:z:0*

DstT0*

SrcT0	*
_output_shapes
:�
8sequential_1/bidirectional_1_2/backward_lstm_1_1/Shape_2Shape@sequential_1/bidirectional_1_2/backward_lstm_1_1/ones_like_1:y:0*
T0*
_output_shapes
::���
Jsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/MulMul@sequential_1/bidirectional_1_2/backward_lstm_1_1/ones_like_1:y:0Ssequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/Const:output:0*
T0*'
_output_shapes
:���������@�
asequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
asequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
zsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter;sequential_1/bidirectional_1_2/backward_lstm_1_1/Cast_1:y:0*
Tseed0* 
_output_shapes
::�
zsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
vsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Asequential_1/bidirectional_1_2/backward_lstm_1_1/Shape_2:output:0�sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0�sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0�sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:���������@�
asequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/subSubjsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/max:output:0jsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
asequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/mulMulsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/StatelessRandomUniformV2:output:0esequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/sub:z:0*
T0*'
_output_shapes
:���������@�
]sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniformAddV2esequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/mul:z:0jsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform/min:output:0*
T0*'
_output_shapes
:���������@�
Ssequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
Qsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/GreaterEqualGreaterEqualasequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/stateless_random_uniform:z:0\sequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
Lsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
Msequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/SelectV2SelectV2Usequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/GreaterEqual:z:0Lsequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/Mul:z:0Usequential_1/bidirectional_1_2/backward_lstm_1_1/stateless_dropout_1/Const_1:output:0*
T0*'
_output_shapes
:���������@�
?sequential_1/bidirectional_1_2/backward_lstm_1_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
:sequential_1/bidirectional_1_2/backward_lstm_1_1/transpose	Transpose6sequential_1/batch_normalization_1/batchnorm/add_1:z:0Hsequential_1/bidirectional_1_2/backward_lstm_1_1/transpose/perm:output:0*
T0*,
_output_shapes
:x�����������
?sequential_1/bidirectional_1_2/backward_lstm_1_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
;sequential_1/bidirectional_1_2/backward_lstm_1_1/ExpandDims
ExpandDimssequential_1/NotEqual_1:z:0Hsequential_1/bidirectional_1_2/backward_lstm_1_1/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:���������x�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
<sequential_1/bidirectional_1_2/backward_lstm_1_1/transpose_1	TransposeDsequential_1/bidirectional_1_2/backward_lstm_1_1/ExpandDims:output:0Jsequential_1/bidirectional_1_2/backward_lstm_1_1/transpose_1/perm:output:0*
T0
*+
_output_shapes
:x����������
Lsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Ksequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
>sequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2TensorListReserveUsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2/element_shape:output:0Tsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
?sequential_1/bidirectional_1_2/backward_lstm_1_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
:sequential_1/bidirectional_1_2/backward_lstm_1_1/ReverseV2	ReverseV2>sequential_1/bidirectional_1_2/backward_lstm_1_1/transpose:y:0Hsequential_1/bidirectional_1_2/backward_lstm_1_1/ReverseV2/axis:output:0*
T0*,
_output_shapes
:x�����������
fsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Xsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorCsequential_1/bidirectional_1_2/backward_lstm_1_1/ReverseV2:output:0osequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
@sequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_2StridedSlice>sequential_1/bidirectional_1_2/backward_lstm_1_1/transpose:y:0Osequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_2/stack:output:0Qsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_2/stack_1:output:0Qsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
Psequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOpYsequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Csequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/MatMulMatMulIsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_2:output:0Xsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Rsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOp[sequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_cast_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
Esequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/MatMul_1MatMul?sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros:output:0Zsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
@sequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/addAddV2Msequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/MatMul:product:0Osequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Qsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOpZsequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Bsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_1AddV2Dsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add:z:0Ysequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Lsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Bsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/splitSplitUsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/split/split_dim:output:0Fsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split�
Dsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/SigmoidSigmoidKsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@�
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Sigmoid_1SigmoidKsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@�
@sequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/mulMulJsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Sigmoid_1:y:0Asequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1:output:0*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/TanhTanhKsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@�
Bsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/mul_1MulHsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Sigmoid:y:0Esequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:���������@�
Bsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_2AddV2Dsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/mul:z:0Fsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@�
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Sigmoid_2SigmoidKsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@�
Csequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Tanh_1TanhFsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:���������@�
Bsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/mul_2MulJsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Sigmoid_2:y:0Gsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
Nsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
Msequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
@sequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_1TensorListReserveWsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_1/element_shape:output:0Vsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���w
5sequential_1/bidirectional_1_2/backward_lstm_1_1/timeConst*
_output_shapes
: *
dtype0*
value	B : }
;sequential_1/bidirectional_1_2/backward_lstm_1_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value	B :xw
5sequential_1/bidirectional_1_2/backward_lstm_1_1/RankConst*
_output_shapes
: *
dtype0*
value	B : ~
<sequential_1/bidirectional_1_2/backward_lstm_1_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : ~
<sequential_1/bidirectional_1_2/backward_lstm_1_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
6sequential_1/bidirectional_1_2/backward_lstm_1_1/rangeRangeEsequential_1/bidirectional_1_2/backward_lstm_1_1/range/start:output:0>sequential_1/bidirectional_1_2/backward_lstm_1_1/Rank:output:0Esequential_1/bidirectional_1_2/backward_lstm_1_1/range/delta:output:0*
_output_shapes
: |
:sequential_1/bidirectional_1_2/backward_lstm_1_1/Max/inputConst*
_output_shapes
: *
dtype0*
value	B :x�
4sequential_1/bidirectional_1_2/backward_lstm_1_1/MaxMaxCsequential_1/bidirectional_1_2/backward_lstm_1_1/Max/input:output:0?sequential_1/bidirectional_1_2/backward_lstm_1_1/range:output:0*
T0*
_output_shapes
: �
Asequential_1/bidirectional_1_2/backward_lstm_1_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: �
<sequential_1/bidirectional_1_2/backward_lstm_1_1/ReverseV2_1	ReverseV2@sequential_1/bidirectional_1_2/backward_lstm_1_1/transpose_1:y:0Jsequential_1/bidirectional_1_2/backward_lstm_1_1/ReverseV2_1/axis:output:0*
T0
*+
_output_shapes
:x����������
Nsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Msequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :x�
@sequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_2TensorListReserveWsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_2/element_shape:output:0Vsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
hsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Zsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorEsequential_1/bidirectional_1_2/backward_lstm_1_1/ReverseV2_1:output:0qsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
;sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_like	ZerosLikeFsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/mul_2:z:0*
T0*'
_output_shapes
:���������@�
Csequential_1/bidirectional_1_2/backward_lstm_1_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
6sequential_1/bidirectional_1_2/backward_lstm_1_1/whileWhileLsequential_1/bidirectional_1_2/backward_lstm_1_1/while/loop_counter:output:0=sequential_1/bidirectional_1_2/backward_lstm_1_1/Max:output:0>sequential_1/bidirectional_1_2/backward_lstm_1_1/time:output:0Isequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2_1:handle:0?sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_like:y:0?sequential_1/bidirectional_1_2/backward_lstm_1_1/zeros:output:0Asequential_1/bidirectional_1_2/backward_lstm_1_1/zeros_1:output:0hsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0jsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Ysequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_cast_readvariableop_resource[sequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_cast_1_readvariableop_resourceZsequential_1_bidirectional_1_2_backward_lstm_1_1_lstm_cell_1_add_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*_
_output_shapesM
K: : : : :���������@:���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*L
bodyDRB
@sequential_1_bidirectional_1_2_backward_lstm_1_1_while_body_2368*L
condDRB
@sequential_1_bidirectional_1_2_backward_lstm_1_1_while_cond_2367*^
output_shapesM
K: : : : :���������@:���������@:���������@: : : : : *
parallel_iterations �
asequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
Ssequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2Stack/TensorListStackTensorListStack?sequential_1/bidirectional_1_2/backward_lstm_1_1/while:output:3jsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elements�
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
@sequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_3StridedSlice\sequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2Stack/TensorListStack:tensor:0Osequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_3/stack:output:0Qsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_3/stack_1:output:0Qsequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
<sequential_1/bidirectional_1_2/backward_lstm_1_1/transpose_2	Transpose\sequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayV2Stack/TensorListStack:tensor:0Jsequential_1/bidirectional_1_2/backward_lstm_1_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:���������@u
*sequential_1/bidirectional_1_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
%sequential_1/bidirectional_1_2/concatConcatV2Hsequential_1/bidirectional_1_2/forward_lstm_1_1/strided_slice_3:output:0Isequential_1/bidirectional_1_2/backward_lstm_1_1/strided_slice_3:output:03sequential_1/bidirectional_1_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
(sequential_1/dense_1/Cast/ReadVariableOpReadVariableOp1sequential_1_dense_1_cast_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_1/dense_1/MatMulMatMul.sequential_1/bidirectional_1_2/concat:output:00sequential_1/dense_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'sequential_1/dense_1/Add/ReadVariableOpReadVariableOp0sequential_1_dense_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_1/dense_1/AddAddV2%sequential_1/dense_1/MatMul:product:0/sequential_1/dense_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@q
sequential_1/dense_1/ReluRelusequential_1/dense_1/Add:z:0*
T0*'
_output_shapes
:���������@�
8sequential_1/batch_normalization_1_2/Cast/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_1_2_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
:sequential_1/batch_normalization_1_2/Cast_1/ReadVariableOpReadVariableOpCsequential_1_batch_normalization_1_2_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
:sequential_1/batch_normalization_1_2/Cast_2/ReadVariableOpReadVariableOpCsequential_1_batch_normalization_1_2_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
:sequential_1/batch_normalization_1_2/Cast_3/ReadVariableOpReadVariableOpCsequential_1_batch_normalization_1_2_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0y
4sequential_1/batch_normalization_1_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2sequential_1/batch_normalization_1_2/batchnorm/addAddV2Bsequential_1/batch_normalization_1_2/Cast_1/ReadVariableOp:value:0=sequential_1/batch_normalization_1_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
4sequential_1/batch_normalization_1_2/batchnorm/RsqrtRsqrt6sequential_1/batch_normalization_1_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2sequential_1/batch_normalization_1_2/batchnorm/mulMul8sequential_1/batch_normalization_1_2/batchnorm/Rsqrt:y:0Bsequential_1/batch_normalization_1_2/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
4sequential_1/batch_normalization_1_2/batchnorm/mul_1Mul'sequential_1/dense_1/Relu:activations:06sequential_1/batch_normalization_1_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
4sequential_1/batch_normalization_1_2/batchnorm/mul_2Mul@sequential_1/batch_normalization_1_2/Cast/ReadVariableOp:value:06sequential_1/batch_normalization_1_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
2sequential_1/batch_normalization_1_2/batchnorm/subSubBsequential_1/batch_normalization_1_2/Cast_3/ReadVariableOp:value:08sequential_1/batch_normalization_1_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
4sequential_1/batch_normalization_1_2/batchnorm/add_1AddV28sequential_1/batch_normalization_1_2/batchnorm/mul_1:z:06sequential_1/batch_normalization_1_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
*sequential_1/dense_1_2/Cast/ReadVariableOpReadVariableOp3sequential_1_dense_1_2_cast_readvariableop_resource*
_output_shapes

:@*
dtype0�
sequential_1/dense_1_2/MatMulMatMul8sequential_1/batch_normalization_1_2/batchnorm/add_1:z:02sequential_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential_1/dense_1_2/Add/ReadVariableOpReadVariableOp2sequential_1_dense_1_2_add_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_1_2/AddAddV2'sequential_1/dense_1_2/MatMul:product:01sequential_1/dense_1_2/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������{
sequential_1/dense_1_2/SoftmaxSoftmaxsequential_1/dense_1_2/Add:z:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential_1/dense_1_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp7^sequential_1/batch_normalization_1/Cast/ReadVariableOp9^sequential_1/batch_normalization_1/Cast_1/ReadVariableOp9^sequential_1/batch_normalization_1/Cast_2/ReadVariableOp9^sequential_1/batch_normalization_1/Cast_3/ReadVariableOp9^sequential_1/batch_normalization_1_2/Cast/ReadVariableOp;^sequential_1/batch_normalization_1_2/Cast_1/ReadVariableOp;^sequential_1/batch_normalization_1_2/Cast_2/ReadVariableOp;^sequential_1/batch_normalization_1_2/Cast_3/ReadVariableOp@^sequential_1/bidirectional_1/backward_lstm_1/Add/ReadVariableOpB^sequential_1/bidirectional_1/backward_lstm_1/Add_1/ReadVariableOp>^sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp@^sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp_1<^sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp>^sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp_1M^sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast/ReadVariableOpO^sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOpN^sequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_1/ReadVariableOp3^sequential_1/bidirectional_1/backward_lstm_1/while?^sequential_1/bidirectional_1/forward_lstm_1/Add/ReadVariableOpA^sequential_1/bidirectional_1/forward_lstm_1/Add_1/ReadVariableOp=^sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp?^sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp_1;^sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp=^sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp_1L^sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast/ReadVariableOpN^sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOpM^sequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_1/ReadVariableOp2^sequential_1/bidirectional_1/forward_lstm_1/whileD^sequential_1/bidirectional_1_2/backward_lstm_1_1/Add/ReadVariableOpF^sequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1/ReadVariableOpB^sequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOpD^sequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp_1@^sequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOpB^sequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp_1Q^sequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOpS^sequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOpR^sequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOp7^sequential_1/bidirectional_1_2/backward_lstm_1_1/whileC^sequential_1/bidirectional_1_2/forward_lstm_1_1/Add/ReadVariableOpE^sequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1/ReadVariableOpA^sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOpC^sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp_1?^sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOpA^sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp_1P^sequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOpR^sequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOpQ^sequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOp6^sequential_1/bidirectional_1_2/forward_lstm_1_1/while(^sequential_1/dense_1/Add/ReadVariableOp)^sequential_1/dense_1/Cast/ReadVariableOp*^sequential_1/dense_1_2/Add/ReadVariableOp+^sequential_1/dense_1_2/Cast/ReadVariableOp1^sequential_1/embedding_1/GatherV2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:���������x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6sequential_1/batch_normalization_1/Cast/ReadVariableOp6sequential_1/batch_normalization_1/Cast/ReadVariableOp2t
8sequential_1/batch_normalization_1/Cast_1/ReadVariableOp8sequential_1/batch_normalization_1/Cast_1/ReadVariableOp2t
8sequential_1/batch_normalization_1/Cast_2/ReadVariableOp8sequential_1/batch_normalization_1/Cast_2/ReadVariableOp2t
8sequential_1/batch_normalization_1/Cast_3/ReadVariableOp8sequential_1/batch_normalization_1/Cast_3/ReadVariableOp2t
8sequential_1/batch_normalization_1_2/Cast/ReadVariableOp8sequential_1/batch_normalization_1_2/Cast/ReadVariableOp2x
:sequential_1/batch_normalization_1_2/Cast_1/ReadVariableOp:sequential_1/batch_normalization_1_2/Cast_1/ReadVariableOp2x
:sequential_1/batch_normalization_1_2/Cast_2/ReadVariableOp:sequential_1/batch_normalization_1_2/Cast_2/ReadVariableOp2x
:sequential_1/batch_normalization_1_2/Cast_3/ReadVariableOp:sequential_1/batch_normalization_1_2/Cast_3/ReadVariableOp2�
?sequential_1/bidirectional_1/backward_lstm_1/Add/ReadVariableOp?sequential_1/bidirectional_1/backward_lstm_1/Add/ReadVariableOp2�
Asequential_1/bidirectional_1/backward_lstm_1/Add_1/ReadVariableOpAsequential_1/bidirectional_1/backward_lstm_1/Add_1/ReadVariableOp2~
=sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp=sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp2�
?sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp_1?sequential_1/bidirectional_1/backward_lstm_1/AssignVariableOp_12z
;sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp;sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp2~
=sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp_1=sequential_1/bidirectional_1/backward_lstm_1/ReadVariableOp_12�
Lsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast/ReadVariableOpLsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast/ReadVariableOp2�
Nsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOpNsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOp2�
Msequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_1/ReadVariableOpMsequential_1/bidirectional_1/backward_lstm_1/lstm_cell_1/add_1/ReadVariableOp2h
2sequential_1/bidirectional_1/backward_lstm_1/while2sequential_1/bidirectional_1/backward_lstm_1/while2�
>sequential_1/bidirectional_1/forward_lstm_1/Add/ReadVariableOp>sequential_1/bidirectional_1/forward_lstm_1/Add/ReadVariableOp2�
@sequential_1/bidirectional_1/forward_lstm_1/Add_1/ReadVariableOp@sequential_1/bidirectional_1/forward_lstm_1/Add_1/ReadVariableOp2|
<sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp<sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp2�
>sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp_1>sequential_1/bidirectional_1/forward_lstm_1/AssignVariableOp_12x
:sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp:sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp2|
<sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp_1<sequential_1/bidirectional_1/forward_lstm_1/ReadVariableOp_12�
Ksequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast/ReadVariableOpKsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast/ReadVariableOp2�
Msequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOpMsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/Cast_1/ReadVariableOp2�
Lsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_1/ReadVariableOpLsequential_1/bidirectional_1/forward_lstm_1/lstm_cell_1/add_1/ReadVariableOp2f
1sequential_1/bidirectional_1/forward_lstm_1/while1sequential_1/bidirectional_1/forward_lstm_1/while2�
Csequential_1/bidirectional_1_2/backward_lstm_1_1/Add/ReadVariableOpCsequential_1/bidirectional_1_2/backward_lstm_1_1/Add/ReadVariableOp2�
Esequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1/ReadVariableOpEsequential_1/bidirectional_1_2/backward_lstm_1_1/Add_1/ReadVariableOp2�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOpAsequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp2�
Csequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp_1Csequential_1/bidirectional_1_2/backward_lstm_1_1/AssignVariableOp_12�
?sequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp?sequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp2�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp_1Asequential_1/bidirectional_1_2/backward_lstm_1_1/ReadVariableOp_12�
Psequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOpPsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOp2�
Rsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOpRsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOp2�
Qsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOpQsequential_1/bidirectional_1_2/backward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOp2p
6sequential_1/bidirectional_1_2/backward_lstm_1_1/while6sequential_1/bidirectional_1_2/backward_lstm_1_1/while2�
Bsequential_1/bidirectional_1_2/forward_lstm_1_1/Add/ReadVariableOpBsequential_1/bidirectional_1_2/forward_lstm_1_1/Add/ReadVariableOp2�
Dsequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1/ReadVariableOpDsequential_1/bidirectional_1_2/forward_lstm_1_1/Add_1/ReadVariableOp2�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp@sequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp2�
Bsequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp_1Bsequential_1/bidirectional_1_2/forward_lstm_1_1/AssignVariableOp_12�
>sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp>sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp2�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp_1@sequential_1/bidirectional_1_2/forward_lstm_1_1/ReadVariableOp_12�
Osequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOpOsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast/ReadVariableOp2�
Qsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOpQsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/Cast_1/ReadVariableOp2�
Psequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOpPsequential_1/bidirectional_1_2/forward_lstm_1_1/lstm_cell_1/add_1/ReadVariableOp2n
5sequential_1/bidirectional_1_2/forward_lstm_1_1/while5sequential_1/bidirectional_1_2/forward_lstm_1_1/while2R
'sequential_1/dense_1/Add/ReadVariableOp'sequential_1/dense_1/Add/ReadVariableOp2T
(sequential_1/dense_1/Cast/ReadVariableOp(sequential_1/dense_1/Cast/ReadVariableOp2V
)sequential_1/dense_1_2/Add/ReadVariableOp)sequential_1/dense_1_2/Add/ReadVariableOp2X
*sequential_1/dense_1_2/Cast/ReadVariableOp*sequential_1/dense_1_2/Cast/ReadVariableOp2d
0sequential_1/embedding_1/GatherV2/ReadVariableOp0sequential_1/embedding_1/GatherV2/ReadVariableOp:T P
'
_output_shapes
:���������x
%
_user_specified_nameinput_layer:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ڑ
�
;sequential_1_bidirectional_1_forward_lstm_1_while_body_1655t
psequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_loop_countere
asequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_maxA
=sequential_1_bidirectional_1_forward_lstm_1_while_placeholderC
?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_1C
?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_2C
?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_3C
?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_4�
�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_tensorlistfromtensor_0�
�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0p
\sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0:
��r
^sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:
��l
]sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0:	�w
ssequential_1_bidirectional_1_forward_lstm_1_while_selectv2_sequential_1_bidirectional_1_forward_lstm_1_zeros_like_0>
:sequential_1_bidirectional_1_forward_lstm_1_while_identity@
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_1@
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_2@
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_3@
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_4@
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_5@
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_6�
�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_tensorlistfromtensor�
�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_1_tensorlistfromtensorn
Zsequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource:
��p
\sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource:
��j
[sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource:	�u
qsequential_1_bidirectional_1_forward_lstm_1_while_selectv2_sequential_1_bidirectional_1_forward_lstm_1_zeros_like��Qsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOp�Ssequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp�Rsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOp�
csequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Usequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_tensorlistfromtensor_0=sequential_1_bidirectional_1_forward_lstm_1_while_placeholderlsequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
esequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Wsequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0=sequential_1_bidirectional_1_forward_lstm_1_while_placeholdernsequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
Qsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOp\sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
Dsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/MatMulMatMul\sequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0Ysequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Ssequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOp^sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
Fsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/MatMul_1MatMul?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_3[sequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Asequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/addAddV2Nsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/MatMul:product:0Psequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Rsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOp]sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
Csequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_1AddV2Esequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add:z:0Zsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Msequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Csequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/splitSplitVsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/split/split_dim:output:0Gsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
Esequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/SigmoidSigmoidLsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
Gsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Sigmoid_1SigmoidLsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
Asequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/mulMulKsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Sigmoid_1:y:0?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_4*
T0*(
_output_shapes
:�����������
Bsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/TanhTanhLsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
Csequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/mul_1MulIsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Sigmoid:y:0Fsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
Csequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_2AddV2Esequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/mul:z:0Gsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
Gsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Sigmoid_2SigmoidLsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
Dsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Tanh_1TanhGsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
Csequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/mul_2MulKsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Sigmoid_2:y:0Hsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
@sequential_1/bidirectional_1/forward_lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
6sequential_1/bidirectional_1/forward_lstm_1/while/TileTile^sequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Isequential_1/bidirectional_1/forward_lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
:sequential_1/bidirectional_1/forward_lstm_1/while/SelectV2SelectV2?sequential_1/bidirectional_1/forward_lstm_1/while/Tile:output:0Gsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/mul_2:z:0ssequential_1_bidirectional_1_forward_lstm_1_while_selectv2_sequential_1_bidirectional_1_forward_lstm_1_zeros_like_0*
T0*(
_output_shapes
:�����������
Bsequential_1/bidirectional_1/forward_lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
8sequential_1/bidirectional_1/forward_lstm_1/while/Tile_1Tile^sequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Ksequential_1/bidirectional_1/forward_lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
Bsequential_1/bidirectional_1/forward_lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
8sequential_1/bidirectional_1/forward_lstm_1/while/Tile_2Tile^sequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Ksequential_1/bidirectional_1/forward_lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:����������
<sequential_1/bidirectional_1/forward_lstm_1/while/SelectV2_1SelectV2Asequential_1/bidirectional_1/forward_lstm_1/while/Tile_1:output:0Gsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/mul_2:z:0?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_3*
T0*(
_output_shapes
:�����������
<sequential_1/bidirectional_1/forward_lstm_1/while/SelectV2_2SelectV2Asequential_1/bidirectional_1/forward_lstm_1/while/Tile_2:output:0Gsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_2:z:0?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_4*
T0*(
_output_shapes
:�����������
Vsequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_1=sequential_1_bidirectional_1_forward_lstm_1_while_placeholderCsequential_1/bidirectional_1/forward_lstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���y
7sequential_1/bidirectional_1/forward_lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
5sequential_1/bidirectional_1/forward_lstm_1/while/addAddV2=sequential_1_bidirectional_1_forward_lstm_1_while_placeholder@sequential_1/bidirectional_1/forward_lstm_1/while/add/y:output:0*
T0*
_output_shapes
: {
9sequential_1/bidirectional_1/forward_lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
7sequential_1/bidirectional_1/forward_lstm_1/while/add_1AddV2psequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_loop_counterBsequential_1/bidirectional_1/forward_lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
:sequential_1/bidirectional_1/forward_lstm_1/while/IdentityIdentity;sequential_1/bidirectional_1/forward_lstm_1/while/add_1:z:07^sequential_1/bidirectional_1/forward_lstm_1/while/NoOp*
T0*
_output_shapes
: �
<sequential_1/bidirectional_1/forward_lstm_1/while/Identity_1Identityasequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_max7^sequential_1/bidirectional_1/forward_lstm_1/while/NoOp*
T0*
_output_shapes
: �
<sequential_1/bidirectional_1/forward_lstm_1/while/Identity_2Identity9sequential_1/bidirectional_1/forward_lstm_1/while/add:z:07^sequential_1/bidirectional_1/forward_lstm_1/while/NoOp*
T0*
_output_shapes
: �
<sequential_1/bidirectional_1/forward_lstm_1/while/Identity_3Identityfsequential_1/bidirectional_1/forward_lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^sequential_1/bidirectional_1/forward_lstm_1/while/NoOp*
T0*
_output_shapes
: �
<sequential_1/bidirectional_1/forward_lstm_1/while/Identity_4IdentityCsequential_1/bidirectional_1/forward_lstm_1/while/SelectV2:output:07^sequential_1/bidirectional_1/forward_lstm_1/while/NoOp*
T0*(
_output_shapes
:�����������
<sequential_1/bidirectional_1/forward_lstm_1/while/Identity_5IdentityEsequential_1/bidirectional_1/forward_lstm_1/while/SelectV2_1:output:07^sequential_1/bidirectional_1/forward_lstm_1/while/NoOp*
T0*(
_output_shapes
:�����������
<sequential_1/bidirectional_1/forward_lstm_1/while/Identity_6IdentityEsequential_1/bidirectional_1/forward_lstm_1/while/SelectV2_2:output:07^sequential_1/bidirectional_1/forward_lstm_1/while/NoOp*
T0*(
_output_shapes
:�����������
6sequential_1/bidirectional_1/forward_lstm_1/while/NoOpNoOpR^sequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOpT^sequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOpS^sequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "�
:sequential_1_bidirectional_1_forward_lstm_1_while_identityCsequential_1/bidirectional_1/forward_lstm_1/while/Identity:output:0"�
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_1Esequential_1/bidirectional_1/forward_lstm_1/while/Identity_1:output:0"�
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_2Esequential_1/bidirectional_1/forward_lstm_1/while/Identity_2:output:0"�
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_3Esequential_1/bidirectional_1/forward_lstm_1/while/Identity_3:output:0"�
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_4Esequential_1/bidirectional_1/forward_lstm_1/while/Identity_4:output:0"�
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_5Esequential_1/bidirectional_1/forward_lstm_1/while/Identity_5:output:0"�
<sequential_1_bidirectional_1_forward_lstm_1_while_identity_6Esequential_1/bidirectional_1/forward_lstm_1/while/Identity_6:output:0"�
[sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource]sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
\sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource^sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
Zsequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource\sequential_1_bidirectional_1_forward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
qsequential_1_bidirectional_1_forward_lstm_1_while_selectv2_sequential_1_bidirectional_1_forward_lstm_1_zeros_likessequential_1_bidirectional_1_forward_lstm_1_while_selectv2_sequential_1_bidirectional_1_forward_lstm_1_zeros_like_0"�
�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_1_tensorlistfromtensor�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"�
�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_tensorlistfromtensor�sequential_1_bidirectional_1_forward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b: : : : :����������:����������:����������: : : : : :����������2�
Qsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOpQsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOp2�
Ssequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOpSsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp2�
Rsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOpRsequential_1/bidirectional_1/forward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOp:v r

_output_shapes
: 
X
_user_specified_name@>sequential_1/bidirectional_1/forward_lstm_1/while/loop_counter:gc

_output_shapes
: 
I
_user_specified_name1/sequential_1/bidirectional_1/forward_lstm_1/Max:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:��

_output_shapes
: 
m
_user_specified_nameUSsequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack/TensorListFromTensor:��

_output_shapes
: 
o
_user_specified_nameWUsequential_1/bidirectional_1/forward_lstm_1/TensorArrayUnstack_1/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:�|
(
_output_shapes
:����������
P
_user_specified_name86sequential_1/bidirectional_1/forward_lstm_1/zeros_like
�
�
+__inference_signature_wrapper___call___2565
input_layer
unknown:���
	unknown_0:	
	unknown_1:
��
	unknown_2:
��
	unknown_3:	�
	unknown_4:	
	unknown_5:
��
	unknown_6:
��
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	

unknown_13:
��

unknown_14:	@�

unknown_15:	�

unknown_16:	

unknown_17:
��

unknown_18:	@�

unknown_19:	�

unknown_20:	�@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *"
fR
__inference___call___2501o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:���������x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������x
%
_user_specified_nameinput_layer:$ 

_user_specified_name2505:$ 

_user_specified_name2507:$ 

_user_specified_name2509:$ 

_user_specified_name2511:$ 

_user_specified_name2513:$ 

_user_specified_name2515:$ 

_user_specified_name2517:$ 

_user_specified_name2519:$	 

_user_specified_name2521:$
 

_user_specified_name2523:$ 

_user_specified_name2525:$ 

_user_specified_name2527:$ 

_user_specified_name2529:$ 

_user_specified_name2531:$ 

_user_specified_name2533:$ 

_user_specified_name2535:$ 

_user_specified_name2537:$ 

_user_specified_name2539:$ 

_user_specified_name2541:$ 

_user_specified_name2543:$ 

_user_specified_name2545:$ 

_user_specified_name2547:$ 

_user_specified_name2549:$ 

_user_specified_name2551:$ 

_user_specified_name2553:$ 

_user_specified_name2555:$ 

_user_specified_name2557:$ 

_user_specified_name2559:$ 

_user_specified_name2561
�
�
?sequential_1_bidirectional_1_2_forward_lstm_1_1_while_cond_2136|
xsequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_loop_counterm
isequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_maxE
Asequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholderG
Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_1G
Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_2G
Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_3G
Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_4�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_cond_2136___redundant_placeholder0�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_cond_2136___redundant_placeholder1�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_cond_2136___redundant_placeholder2�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_cond_2136___redundant_placeholder3�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_cond_2136___redundant_placeholder4B
>sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity
~
<sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :x�
:sequential_1/bidirectional_1_2/forward_lstm_1_1/while/LessLessAsequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholderEsequential_1/bidirectional_1_2/forward_lstm_1_1/while/Less/y:output:0*
T0*
_output_shapes
: �
<sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Less_1Lessxsequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_loop_counterisequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_max*
T0*
_output_shapes
: �
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/LogicalAnd
LogicalAnd@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Less_1:z:0>sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Less:z:0*
_output_shapes
: �
>sequential_1/bidirectional_1_2/forward_lstm_1_1/while/IdentityIdentityDsequential_1/bidirectional_1_2/forward_lstm_1_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "�
>sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identityGsequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U: : : : :���������@:���������@:���������@::::::z v

_output_shapes
: 
\
_user_specified_nameDBsequential_1/bidirectional_1_2/forward_lstm_1_1/while/loop_counter:kg

_output_shapes
: 
M
_user_specified_name53sequential_1/bidirectional_1_2/forward_lstm_1_1/Max:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
::

_output_shapes
:
��
�'
 __inference__traced_restore_3325
file_prefix1
assignvariableop_variable_30:���,
assignvariableop_1_variable_29:	2
assignvariableop_2_variable_28:
��2
assignvariableop_3_variable_27:
��-
assignvariableop_4_variable_26:	�,
assignvariableop_5_variable_25:	2
assignvariableop_6_variable_24:
��2
assignvariableop_7_variable_23:
��-
assignvariableop_8_variable_22:	�,
assignvariableop_9_variable_21:	.
assignvariableop_10_variable_20:	�.
assignvariableop_11_variable_19:	�.
assignvariableop_12_variable_18:	�.
assignvariableop_13_variable_17:	�3
assignvariableop_14_variable_16:
��2
assignvariableop_15_variable_15:	@�.
assignvariableop_16_variable_14:	�-
assignvariableop_17_variable_13:	3
assignvariableop_18_variable_12:
��2
assignvariableop_19_variable_11:	@�.
assignvariableop_20_variable_10:	�,
assignvariableop_21_variable_9:	1
assignvariableop_22_variable_8:	�@,
assignvariableop_23_variable_7:@,
assignvariableop_24_variable_6:@,
assignvariableop_25_variable_5:@,
assignvariableop_26_variable_4:@,
assignvariableop_27_variable_3:@,
assignvariableop_28_variable_2:	0
assignvariableop_29_variable_1:@*
assignvariableop_30_variable:@
1assignvariableop_31_forward_lstm_lstm_cell_bias_1:	�I
:assignvariableop_32_sequential_batch_normalization_gamma_1:	�J
<assignvariableop_33_sequential_batch_normalization_1_gamma_1:@J
5assignvariableop_34_sequential_embedding_embeddings_1:���Q
=assignvariableop_35_forward_lstm_lstm_cell_recurrent_kernel_1:
��J
6assignvariableop_36_backward_lstm_1_lstm_cell_kernel_1:
��A
2assignvariableop_37_backward_lstm_lstm_cell_bias_1:	�B
3assignvariableop_38_forward_lstm_1_lstm_cell_bias_1:	�R
>assignvariableop_39_backward_lstm_lstm_cell_recurrent_kernel_1:
��I
;assignvariableop_40_sequential_batch_normalization_1_beta_1:@;
-assignvariableop_41_sequential_dense_1_bias_1:R
?assignvariableop_42_forward_lstm_1_lstm_cell_recurrent_kernel_1:	@�A
/assignvariableop_43_sequential_dense_1_kernel_1:@G
3assignvariableop_44_forward_lstm_lstm_cell_kernel_1:
��S
@assignvariableop_45_backward_lstm_1_lstm_cell_recurrent_kernel_1:	@�I
5assignvariableop_46_forward_lstm_1_lstm_cell_kernel_1:
��C
4assignvariableop_47_backward_lstm_1_lstm_cell_bias_1:	�9
+assignvariableop_48_sequential_dense_bias_1:@H
4assignvariableop_49_backward_lstm_lstm_cell_kernel_1:
��H
9assignvariableop_50_sequential_batch_normalization_beta_1:	�@
-assignvariableop_51_sequential_dense_kernel_1:	�@I
;assignvariableop_52_seed_generator_2_seed_generator_state_1:	O
@assignvariableop_53_sequential_batch_normalization_moving_mean_1:	�T
Fassignvariableop_54_sequential_batch_normalization_1_moving_variance_1:@I
;assignvariableop_55_seed_generator_6_seed_generator_state_1:	I
;assignvariableop_56_seed_generator_5_seed_generator_state_1:	S
Dassignvariableop_57_sequential_batch_normalization_moving_variance_1:	�P
Bassignvariableop_58_sequential_batch_normalization_1_moving_mean_1:@I
;assignvariableop_59_seed_generator_3_seed_generator_state_1:	
identity_61��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=										[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_30Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_29Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_28Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_27Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_26Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_25Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_24Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_23Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_22Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_21Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_20Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_19Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_18Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_17Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_16Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_15Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_14Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_13Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_12Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_11Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_10Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_9Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_8Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_7Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_6Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_5Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_4Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variable_3Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_variable_2Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_variable_1Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_variableIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp1assignvariableop_31_forward_lstm_lstm_cell_bias_1Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp:assignvariableop_32_sequential_batch_normalization_gamma_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp<assignvariableop_33_sequential_batch_normalization_1_gamma_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp5assignvariableop_34_sequential_embedding_embeddings_1Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp=assignvariableop_35_forward_lstm_lstm_cell_recurrent_kernel_1Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp6assignvariableop_36_backward_lstm_1_lstm_cell_kernel_1Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp2assignvariableop_37_backward_lstm_lstm_cell_bias_1Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp3assignvariableop_38_forward_lstm_1_lstm_cell_bias_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp>assignvariableop_39_backward_lstm_lstm_cell_recurrent_kernel_1Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp;assignvariableop_40_sequential_batch_normalization_1_beta_1Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp-assignvariableop_41_sequential_dense_1_bias_1Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp?assignvariableop_42_forward_lstm_1_lstm_cell_recurrent_kernel_1Identity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp/assignvariableop_43_sequential_dense_1_kernel_1Identity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp3assignvariableop_44_forward_lstm_lstm_cell_kernel_1Identity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp@assignvariableop_45_backward_lstm_1_lstm_cell_recurrent_kernel_1Identity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp5assignvariableop_46_forward_lstm_1_lstm_cell_kernel_1Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp4assignvariableop_47_backward_lstm_1_lstm_cell_bias_1Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp+assignvariableop_48_sequential_dense_bias_1Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp4assignvariableop_49_backward_lstm_lstm_cell_kernel_1Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp9assignvariableop_50_sequential_batch_normalization_beta_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp-assignvariableop_51_sequential_dense_kernel_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp;assignvariableop_52_seed_generator_2_seed_generator_state_1Identity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp@assignvariableop_53_sequential_batch_normalization_moving_mean_1Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpFassignvariableop_54_sequential_batch_normalization_1_moving_variance_1Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp;assignvariableop_55_seed_generator_6_seed_generator_state_1Identity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp;assignvariableop_56_seed_generator_5_seed_generator_state_1Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpDassignvariableop_57_sequential_batch_normalization_moving_variance_1Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpBassignvariableop_58_sequential_batch_normalization_1_moving_mean_1Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp;assignvariableop_59_seed_generator_3_seed_generator_state_1Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_61IdentityIdentity_60:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_61Identity_61:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+'
%
_user_specified_nameVariable_30:+'
%
_user_specified_nameVariable_29:+'
%
_user_specified_nameVariable_28:+'
%
_user_specified_nameVariable_27:+'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_23:+	'
%
_user_specified_nameVariable_22:+
'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_10:*&
$
_user_specified_name
Variable_9:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_1:($
"
_user_specified_name
Variable:= 9
7
_user_specified_nameforward_lstm/lstm_cell/bias_1:F!B
@
_user_specified_name(&sequential/batch_normalization/gamma_1:H"D
B
_user_specified_name*(sequential/batch_normalization_1/gamma_1:A#=
;
_user_specified_name#!sequential/embedding/embeddings_1:I$E
C
_user_specified_name+)forward_lstm/lstm_cell/recurrent_kernel_1:B%>
<
_user_specified_name$"backward_lstm_1/lstm_cell/kernel_1:>&:
8
_user_specified_name backward_lstm/lstm_cell/bias_1:?';
9
_user_specified_name!forward_lstm_1/lstm_cell/bias_1:J(F
D
_user_specified_name,*backward_lstm/lstm_cell/recurrent_kernel_1:G)C
A
_user_specified_name)'sequential/batch_normalization_1/beta_1:9*5
3
_user_specified_namesequential/dense_1/bias_1:K+G
E
_user_specified_name-+forward_lstm_1/lstm_cell/recurrent_kernel_1:;,7
5
_user_specified_namesequential/dense_1/kernel_1:?-;
9
_user_specified_name!forward_lstm/lstm_cell/kernel_1:L.H
F
_user_specified_name.,backward_lstm_1/lstm_cell/recurrent_kernel_1:A/=
;
_user_specified_name#!forward_lstm_1/lstm_cell/kernel_1:@0<
:
_user_specified_name" backward_lstm_1/lstm_cell/bias_1:713
1
_user_specified_namesequential/dense/bias_1:@2<
:
_user_specified_name" backward_lstm/lstm_cell/kernel_1:E3A
?
_user_specified_name'%sequential/batch_normalization/beta_1:945
3
_user_specified_namesequential/dense/kernel_1:G5C
A
_user_specified_name)'seed_generator_2/seed_generator_state_1:L6H
F
_user_specified_name.,sequential/batch_normalization/moving_mean_1:R7N
L
_user_specified_name42sequential/batch_normalization_1/moving_variance_1:G8C
A
_user_specified_name)'seed_generator_6/seed_generator_state_1:G9C
A
_user_specified_name)'seed_generator_5/seed_generator_state_1:P:L
J
_user_specified_name20sequential/batch_normalization/moving_variance_1:N;J
H
_user_specified_name0.sequential/batch_normalization_1/moving_mean_1:G<C
A
_user_specified_name)'seed_generator_3/seed_generator_state_1
�
�
;sequential_1_bidirectional_1_forward_lstm_1_while_cond_1654t
psequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_loop_countere
asequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_maxA
=sequential_1_bidirectional_1_forward_lstm_1_while_placeholderC
?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_1C
?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_2C
?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_3C
?sequential_1_bidirectional_1_forward_lstm_1_while_placeholder_4�
�sequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_cond_1654___redundant_placeholder0�
�sequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_cond_1654___redundant_placeholder1�
�sequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_cond_1654___redundant_placeholder2�
�sequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_cond_1654___redundant_placeholder3�
�sequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_cond_1654___redundant_placeholder4�
�sequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_cond_1654___redundant_placeholder5>
:sequential_1_bidirectional_1_forward_lstm_1_while_identity
z
8sequential_1/bidirectional_1/forward_lstm_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :x�
6sequential_1/bidirectional_1/forward_lstm_1/while/LessLess=sequential_1_bidirectional_1_forward_lstm_1_while_placeholderAsequential_1/bidirectional_1/forward_lstm_1/while/Less/y:output:0*
T0*
_output_shapes
: �
8sequential_1/bidirectional_1/forward_lstm_1/while/Less_1Lesspsequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_while_loop_counterasequential_1_bidirectional_1_forward_lstm_1_while_sequential_1_bidirectional_1_forward_lstm_1_max*
T0*
_output_shapes
: �
<sequential_1/bidirectional_1/forward_lstm_1/while/LogicalAnd
LogicalAnd<sequential_1/bidirectional_1/forward_lstm_1/while/Less_1:z:0:sequential_1/bidirectional_1/forward_lstm_1/while/Less:z:0*
_output_shapes
: �
:sequential_1/bidirectional_1/forward_lstm_1/while/IdentityIdentity@sequential_1/bidirectional_1/forward_lstm_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "�
:sequential_1_bidirectional_1_forward_lstm_1_while_identityCsequential_1/bidirectional_1/forward_lstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\: : : : :����������:����������:����������:::::::v r

_output_shapes
: 
X
_user_specified_name@>sequential_1/bidirectional_1/forward_lstm_1/while/loop_counter:gc

_output_shapes
: 
I
_user_specified_name1/sequential_1/bidirectional_1/forward_lstm_1/Max:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
::

_output_shapes
::

_output_shapes
:
�
�
@sequential_1_bidirectional_1_2_backward_lstm_1_1_while_cond_2367~
zsequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_loop_countero
ksequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_maxF
Bsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholderH
Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_1H
Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_2H
Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_3H
Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_4�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_cond_2367___redundant_placeholder0�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_cond_2367___redundant_placeholder1�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_cond_2367___redundant_placeholder2�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_cond_2367___redundant_placeholder3�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_cond_2367___redundant_placeholder4C
?sequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity

=sequential_1/bidirectional_1_2/backward_lstm_1_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :x�
;sequential_1/bidirectional_1_2/backward_lstm_1_1/while/LessLessBsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholderFsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Less/y:output:0*
T0*
_output_shapes
: �
=sequential_1/bidirectional_1_2/backward_lstm_1_1/while/Less_1Lesszsequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_loop_counterksequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_max*
T0*
_output_shapes
: �
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/LogicalAnd
LogicalAndAsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Less_1:z:0?sequential_1/bidirectional_1_2/backward_lstm_1_1/while/Less:z:0*
_output_shapes
: �
?sequential_1/bidirectional_1_2/backward_lstm_1_1/while/IdentityIdentityEsequential_1/bidirectional_1_2/backward_lstm_1_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "�
?sequential_1_bidirectional_1_2_backward_lstm_1_1_while_identityHsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U: : : : :���������@:���������@:���������@::::::{ w

_output_shapes
: 
]
_user_specified_nameECsequential_1/bidirectional_1_2/backward_lstm_1_1/while/loop_counter:lh

_output_shapes
: 
N
_user_specified_name64sequential_1/bidirectional_1_2/backward_lstm_1_1/Max:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
::

_output_shapes
:
ǲ
�8
__inference__traced_save_3136
file_prefix7
"read_disablecopyonread_variable_30:���2
$read_1_disablecopyonread_variable_29:	8
$read_2_disablecopyonread_variable_28:
��8
$read_3_disablecopyonread_variable_27:
��3
$read_4_disablecopyonread_variable_26:	�2
$read_5_disablecopyonread_variable_25:	8
$read_6_disablecopyonread_variable_24:
��8
$read_7_disablecopyonread_variable_23:
��3
$read_8_disablecopyonread_variable_22:	�2
$read_9_disablecopyonread_variable_21:	4
%read_10_disablecopyonread_variable_20:	�4
%read_11_disablecopyonread_variable_19:	�4
%read_12_disablecopyonread_variable_18:	�4
%read_13_disablecopyonread_variable_17:	�9
%read_14_disablecopyonread_variable_16:
��8
%read_15_disablecopyonread_variable_15:	@�4
%read_16_disablecopyonread_variable_14:	�3
%read_17_disablecopyonread_variable_13:	9
%read_18_disablecopyonread_variable_12:
��8
%read_19_disablecopyonread_variable_11:	@�4
%read_20_disablecopyonread_variable_10:	�2
$read_21_disablecopyonread_variable_9:	7
$read_22_disablecopyonread_variable_8:	�@2
$read_23_disablecopyonread_variable_7:@2
$read_24_disablecopyonread_variable_6:@2
$read_25_disablecopyonread_variable_5:@2
$read_26_disablecopyonread_variable_4:@2
$read_27_disablecopyonread_variable_3:@2
$read_28_disablecopyonread_variable_2:	6
$read_29_disablecopyonread_variable_1:@0
"read_30_disablecopyonread_variable:F
7read_31_disablecopyonread_forward_lstm_lstm_cell_bias_1:	�O
@read_32_disablecopyonread_sequential_batch_normalization_gamma_1:	�P
Bread_33_disablecopyonread_sequential_batch_normalization_1_gamma_1:@P
;read_34_disablecopyonread_sequential_embedding_embeddings_1:���W
Cread_35_disablecopyonread_forward_lstm_lstm_cell_recurrent_kernel_1:
��P
<read_36_disablecopyonread_backward_lstm_1_lstm_cell_kernel_1:
��G
8read_37_disablecopyonread_backward_lstm_lstm_cell_bias_1:	�H
9read_38_disablecopyonread_forward_lstm_1_lstm_cell_bias_1:	�X
Dread_39_disablecopyonread_backward_lstm_lstm_cell_recurrent_kernel_1:
��O
Aread_40_disablecopyonread_sequential_batch_normalization_1_beta_1:@A
3read_41_disablecopyonread_sequential_dense_1_bias_1:X
Eread_42_disablecopyonread_forward_lstm_1_lstm_cell_recurrent_kernel_1:	@�G
5read_43_disablecopyonread_sequential_dense_1_kernel_1:@M
9read_44_disablecopyonread_forward_lstm_lstm_cell_kernel_1:
��Y
Fread_45_disablecopyonread_backward_lstm_1_lstm_cell_recurrent_kernel_1:	@�O
;read_46_disablecopyonread_forward_lstm_1_lstm_cell_kernel_1:
��I
:read_47_disablecopyonread_backward_lstm_1_lstm_cell_bias_1:	�?
1read_48_disablecopyonread_sequential_dense_bias_1:@N
:read_49_disablecopyonread_backward_lstm_lstm_cell_kernel_1:
��N
?read_50_disablecopyonread_sequential_batch_normalization_beta_1:	�F
3read_51_disablecopyonread_sequential_dense_kernel_1:	�@O
Aread_52_disablecopyonread_seed_generator_2_seed_generator_state_1:	U
Fread_53_disablecopyonread_sequential_batch_normalization_moving_mean_1:	�Z
Lread_54_disablecopyonread_sequential_batch_normalization_1_moving_variance_1:@O
Aread_55_disablecopyonread_seed_generator_6_seed_generator_state_1:	O
Aread_56_disablecopyonread_seed_generator_5_seed_generator_state_1:	Y
Jread_57_disablecopyonread_sequential_batch_normalization_moving_variance_1:	�V
Hread_58_disablecopyonread_sequential_batch_normalization_1_moving_mean_1:@O
Aread_59_disablecopyonread_seed_generator_3_seed_generator_state_1:	
savev2_const
identity_121��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_30*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_30^Read/DisableCopyOnRead*!
_output_shapes
:���*
dtype0]
IdentityIdentityRead/ReadVariableOp:value:0*
T0*!
_output_shapes
:���d

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*!
_output_shapes
:���i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_29*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_29^Read_1/DisableCopyOnRead*
_output_shapes
:*
dtype0	Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0	*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0	*
_output_shapes
:i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_28*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_28^Read_2/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0`

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_27*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_27^Read_3/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0`

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��e

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_26*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_26^Read_4/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_25*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_25^Read_5/DisableCopyOnRead*
_output_shapes
:*
dtype0	[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0	*
_output_shapes
:i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_24*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_24^Read_6/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_23*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_23^Read_7/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_22*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_22^Read_8/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_21*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_21^Read_9/DisableCopyOnRead*
_output_shapes
:*
dtype0	[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0	*
_output_shapes
:k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_20*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_20^Read_10/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_19*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_19^Read_11/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_18*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_18^Read_12/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_17*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_17^Read_13/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_16*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_16^Read_14/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_15*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_15^Read_15/DisableCopyOnRead*
_output_shapes
:	@�*
dtype0a
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	@�f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	@�k
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_variable_14*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_variable_14^Read_16/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_17/DisableCopyOnReadDisableCopyOnRead%read_17_disablecopyonread_variable_13*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp%read_17_disablecopyonread_variable_13^Read_17/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0	*
_output_shapes
:k
Read_18/DisableCopyOnReadDisableCopyOnRead%read_18_disablecopyonread_variable_12*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp%read_18_disablecopyonread_variable_12^Read_18/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_variable_11*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_variable_11^Read_19/DisableCopyOnRead*
_output_shapes
:	@�*
dtype0a
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes
:	@�f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	@�k
Read_20/DisableCopyOnReadDisableCopyOnRead%read_20_disablecopyonread_variable_10*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp%read_20_disablecopyonread_variable_10^Read_20/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_21/DisableCopyOnReadDisableCopyOnRead$read_21_disablecopyonread_variable_9*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp$read_21_disablecopyonread_variable_9^Read_21/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0	*
_output_shapes
:j
Read_22/DisableCopyOnReadDisableCopyOnRead$read_22_disablecopyonread_variable_8*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp$read_22_disablecopyonread_variable_8^Read_22/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0a
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@j
Read_23/DisableCopyOnReadDisableCopyOnRead$read_23_disablecopyonread_variable_7*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp$read_23_disablecopyonread_variable_7^Read_23/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_24/DisableCopyOnReadDisableCopyOnRead$read_24_disablecopyonread_variable_6*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp$read_24_disablecopyonread_variable_6^Read_24/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_25/DisableCopyOnReadDisableCopyOnRead$read_25_disablecopyonread_variable_5*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp$read_25_disablecopyonread_variable_5^Read_25/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_26/DisableCopyOnReadDisableCopyOnRead$read_26_disablecopyonread_variable_4*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp$read_26_disablecopyonread_variable_4^Read_26/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_27/DisableCopyOnReadDisableCopyOnRead$read_27_disablecopyonread_variable_3*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp$read_27_disablecopyonread_variable_3^Read_27/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_28/DisableCopyOnReadDisableCopyOnRead$read_28_disablecopyonread_variable_2*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp$read_28_disablecopyonread_variable_2^Read_28/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0	*
_output_shapes
:j
Read_29/DisableCopyOnReadDisableCopyOnRead$read_29_disablecopyonread_variable_1*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp$read_29_disablecopyonread_variable_1^Read_29/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:@h
Read_30/DisableCopyOnReadDisableCopyOnRead"read_30_disablecopyonread_variable*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp"read_30_disablecopyonread_variable^Read_30/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_31/DisableCopyOnReadDisableCopyOnRead7read_31_disablecopyonread_forward_lstm_lstm_cell_bias_1*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp7read_31_disablecopyonread_forward_lstm_lstm_cell_bias_1^Read_31/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_32/DisableCopyOnReadDisableCopyOnRead@read_32_disablecopyonread_sequential_batch_normalization_gamma_1*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp@read_32_disablecopyonread_sequential_batch_normalization_gamma_1^Read_32/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnReadBread_33_disablecopyonread_sequential_batch_normalization_1_gamma_1*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpBread_33_disablecopyonread_sequential_batch_normalization_1_gamma_1^Read_33/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_34/DisableCopyOnReadDisableCopyOnRead;read_34_disablecopyonread_sequential_embedding_embeddings_1*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp;read_34_disablecopyonread_sequential_embedding_embeddings_1^Read_34/DisableCopyOnRead*!
_output_shapes
:���*
dtype0c
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0*!
_output_shapes
:���h
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*!
_output_shapes
:����
Read_35/DisableCopyOnReadDisableCopyOnReadCread_35_disablecopyonread_forward_lstm_lstm_cell_recurrent_kernel_1*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpCread_35_disablecopyonread_forward_lstm_lstm_cell_recurrent_kernel_1^Read_35/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_36/DisableCopyOnReadDisableCopyOnRead<read_36_disablecopyonread_backward_lstm_1_lstm_cell_kernel_1*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp<read_36_disablecopyonread_backward_lstm_1_lstm_cell_kernel_1^Read_36/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_72IdentityRead_36/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��~
Read_37/DisableCopyOnReadDisableCopyOnRead8read_37_disablecopyonread_backward_lstm_lstm_cell_bias_1*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp8read_37_disablecopyonread_backward_lstm_lstm_cell_bias_1^Read_37/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_74IdentityRead_37/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_38/DisableCopyOnReadDisableCopyOnRead9read_38_disablecopyonread_forward_lstm_1_lstm_cell_bias_1*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp9read_38_disablecopyonread_forward_lstm_1_lstm_cell_bias_1^Read_38/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_76IdentityRead_38/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_39/DisableCopyOnReadDisableCopyOnReadDread_39_disablecopyonread_backward_lstm_lstm_cell_recurrent_kernel_1*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpDread_39_disablecopyonread_backward_lstm_lstm_cell_recurrent_kernel_1^Read_39/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_78IdentityRead_39/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_40/DisableCopyOnReadDisableCopyOnReadAread_40_disablecopyonread_sequential_batch_normalization_1_beta_1*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpAread_40_disablecopyonread_sequential_batch_normalization_1_beta_1^Read_40/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_80IdentityRead_40/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_41/DisableCopyOnReadDisableCopyOnRead3read_41_disablecopyonread_sequential_dense_1_bias_1*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp3read_41_disablecopyonread_sequential_dense_1_bias_1^Read_41/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_82IdentityRead_41/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_42/DisableCopyOnReadDisableCopyOnReadEread_42_disablecopyonread_forward_lstm_1_lstm_cell_recurrent_kernel_1*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpEread_42_disablecopyonread_forward_lstm_1_lstm_cell_recurrent_kernel_1^Read_42/DisableCopyOnRead*
_output_shapes
:	@�*
dtype0a
Identity_84IdentityRead_42/ReadVariableOp:value:0*
T0*
_output_shapes
:	@�f
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:	@�{
Read_43/DisableCopyOnReadDisableCopyOnRead5read_43_disablecopyonread_sequential_dense_1_kernel_1*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp5read_43_disablecopyonread_sequential_dense_1_kernel_1^Read_43/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_86IdentityRead_43/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_44/DisableCopyOnReadDisableCopyOnRead9read_44_disablecopyonread_forward_lstm_lstm_cell_kernel_1*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp9read_44_disablecopyonread_forward_lstm_lstm_cell_kernel_1^Read_44/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_88IdentityRead_44/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_45/DisableCopyOnReadDisableCopyOnReadFread_45_disablecopyonread_backward_lstm_1_lstm_cell_recurrent_kernel_1*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpFread_45_disablecopyonread_backward_lstm_1_lstm_cell_recurrent_kernel_1^Read_45/DisableCopyOnRead*
_output_shapes
:	@�*
dtype0a
Identity_90IdentityRead_45/ReadVariableOp:value:0*
T0*
_output_shapes
:	@�f
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:	@��
Read_46/DisableCopyOnReadDisableCopyOnRead;read_46_disablecopyonread_forward_lstm_1_lstm_cell_kernel_1*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp;read_46_disablecopyonread_forward_lstm_1_lstm_cell_kernel_1^Read_46/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_92IdentityRead_46/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_47/DisableCopyOnReadDisableCopyOnRead:read_47_disablecopyonread_backward_lstm_1_lstm_cell_bias_1*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp:read_47_disablecopyonread_backward_lstm_1_lstm_cell_bias_1^Read_47/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_94IdentityRead_47/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes	
:�w
Read_48/DisableCopyOnReadDisableCopyOnRead1read_48_disablecopyonread_sequential_dense_bias_1*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp1read_48_disablecopyonread_sequential_dense_bias_1^Read_48/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_96IdentityRead_48/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_49/DisableCopyOnReadDisableCopyOnRead:read_49_disablecopyonread_backward_lstm_lstm_cell_kernel_1*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp:read_49_disablecopyonread_backward_lstm_lstm_cell_kernel_1^Read_49/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_98IdentityRead_49/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_50/DisableCopyOnReadDisableCopyOnRead?read_50_disablecopyonread_sequential_batch_normalization_beta_1*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp?read_50_disablecopyonread_sequential_batch_normalization_beta_1^Read_50/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_100IdentityRead_50/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:�y
Read_51/DisableCopyOnReadDisableCopyOnRead3read_51_disablecopyonread_sequential_dense_kernel_1*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp3read_51_disablecopyonread_sequential_dense_kernel_1^Read_51/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0b
Identity_102IdentityRead_51/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@h
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_52/DisableCopyOnReadDisableCopyOnReadAread_52_disablecopyonread_seed_generator_2_seed_generator_state_1*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpAread_52_disablecopyonread_seed_generator_2_seed_generator_state_1^Read_52/DisableCopyOnRead*
_output_shapes
:*
dtype0	]
Identity_104IdentityRead_52/ReadVariableOp:value:0*
T0	*
_output_shapes
:c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0	*
_output_shapes
:�
Read_53/DisableCopyOnReadDisableCopyOnReadFread_53_disablecopyonread_sequential_batch_normalization_moving_mean_1*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpFread_53_disablecopyonread_sequential_batch_normalization_moving_mean_1^Read_53/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_106IdentityRead_53/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_54/DisableCopyOnReadDisableCopyOnReadLread_54_disablecopyonread_sequential_batch_normalization_1_moving_variance_1*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpLread_54_disablecopyonread_sequential_batch_normalization_1_moving_variance_1^Read_54/DisableCopyOnRead*
_output_shapes
:@*
dtype0]
Identity_108IdentityRead_54/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_55/DisableCopyOnReadDisableCopyOnReadAread_55_disablecopyonread_seed_generator_6_seed_generator_state_1*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpAread_55_disablecopyonread_seed_generator_6_seed_generator_state_1^Read_55/DisableCopyOnRead*
_output_shapes
:*
dtype0	]
Identity_110IdentityRead_55/ReadVariableOp:value:0*
T0	*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0	*
_output_shapes
:�
Read_56/DisableCopyOnReadDisableCopyOnReadAread_56_disablecopyonread_seed_generator_5_seed_generator_state_1*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpAread_56_disablecopyonread_seed_generator_5_seed_generator_state_1^Read_56/DisableCopyOnRead*
_output_shapes
:*
dtype0	]
Identity_112IdentityRead_56/ReadVariableOp:value:0*
T0	*
_output_shapes
:c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0	*
_output_shapes
:�
Read_57/DisableCopyOnReadDisableCopyOnReadJread_57_disablecopyonread_sequential_batch_normalization_moving_variance_1*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpJread_57_disablecopyonread_sequential_batch_normalization_moving_variance_1^Read_57/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_114IdentityRead_57/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_58/DisableCopyOnReadDisableCopyOnReadHread_58_disablecopyonread_sequential_batch_normalization_1_moving_mean_1*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpHread_58_disablecopyonread_sequential_batch_normalization_1_moving_mean_1^Read_58/DisableCopyOnRead*
_output_shapes
:@*
dtype0]
Identity_116IdentityRead_58/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_59/DisableCopyOnReadDisableCopyOnReadAread_59_disablecopyonread_seed_generator_3_seed_generator_state_1*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpAread_59_disablecopyonread_seed_generator_3_seed_generator_state_1^Read_59/DisableCopyOnRead*
_output_shapes
:*
dtype0	]
Identity_118IdentityRead_59/ReadVariableOp:value:0*
T0	*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0	*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *K
dtypesA
?2=										�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_120Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_121IdentityIdentity_120:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_121Identity_121:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+'
%
_user_specified_nameVariable_30:+'
%
_user_specified_nameVariable_29:+'
%
_user_specified_nameVariable_28:+'
%
_user_specified_nameVariable_27:+'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_23:+	'
%
_user_specified_nameVariable_22:+
'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_10:*&
$
_user_specified_name
Variable_9:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_1:($
"
_user_specified_name
Variable:= 9
7
_user_specified_nameforward_lstm/lstm_cell/bias_1:F!B
@
_user_specified_name(&sequential/batch_normalization/gamma_1:H"D
B
_user_specified_name*(sequential/batch_normalization_1/gamma_1:A#=
;
_user_specified_name#!sequential/embedding/embeddings_1:I$E
C
_user_specified_name+)forward_lstm/lstm_cell/recurrent_kernel_1:B%>
<
_user_specified_name$"backward_lstm_1/lstm_cell/kernel_1:>&:
8
_user_specified_name backward_lstm/lstm_cell/bias_1:?';
9
_user_specified_name!forward_lstm_1/lstm_cell/bias_1:J(F
D
_user_specified_name,*backward_lstm/lstm_cell/recurrent_kernel_1:G)C
A
_user_specified_name)'sequential/batch_normalization_1/beta_1:9*5
3
_user_specified_namesequential/dense_1/bias_1:K+G
E
_user_specified_name-+forward_lstm_1/lstm_cell/recurrent_kernel_1:;,7
5
_user_specified_namesequential/dense_1/kernel_1:?-;
9
_user_specified_name!forward_lstm/lstm_cell/kernel_1:L.H
F
_user_specified_name.,backward_lstm_1/lstm_cell/recurrent_kernel_1:A/=
;
_user_specified_name#!forward_lstm_1/lstm_cell/kernel_1:@0<
:
_user_specified_name" backward_lstm_1/lstm_cell/bias_1:713
1
_user_specified_namesequential/dense/bias_1:@2<
:
_user_specified_name" backward_lstm/lstm_cell/kernel_1:E3A
?
_user_specified_name'%sequential/batch_normalization/beta_1:945
3
_user_specified_namesequential/dense/kernel_1:G5C
A
_user_specified_name)'seed_generator_2/seed_generator_state_1:L6H
F
_user_specified_name.,sequential/batch_normalization/moving_mean_1:R7N
L
_user_specified_name42sequential/batch_normalization_1/moving_variance_1:G8C
A
_user_specified_name)'seed_generator_6/seed_generator_state_1:G9C
A
_user_specified_name)'seed_generator_5/seed_generator_state_1:P:L
J
_user_specified_name20sequential/batch_normalization/moving_variance_1:N;J
H
_user_specified_name0.sequential/batch_normalization_1/moving_mean_1:G<C
A
_user_specified_name)'seed_generator_3/seed_generator_state_1:==9

_output_shapes
: 

_user_specified_nameConst
��
�
<sequential_1_bidirectional_1_backward_lstm_1_while_body_1888v
rsequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_loop_counterg
csequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_maxB
>sequential_1_bidirectional_1_backward_lstm_1_while_placeholderD
@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_1D
@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_2D
@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_3D
@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_4�
�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_tensorlistfromtensor_0�
�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0q
]sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0:
��s
_sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:
��m
^sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0:	�y
usequential_1_bidirectional_1_backward_lstm_1_while_selectv2_sequential_1_bidirectional_1_backward_lstm_1_zeros_like_0?
;sequential_1_bidirectional_1_backward_lstm_1_while_identityA
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_1A
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_2A
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_3A
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_4A
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_5A
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_6�
�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_tensorlistfromtensor�
�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_1_tensorlistfromtensoro
[sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource:
��q
]sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource:
��k
\sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource:	�w
ssequential_1_bidirectional_1_backward_lstm_1_while_selectv2_sequential_1_bidirectional_1_backward_lstm_1_zeros_like��Rsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOp�Tsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp�Ssequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOp�
dsequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Vsequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_tensorlistfromtensor_0>sequential_1_bidirectional_1_backward_lstm_1_while_placeholdermsequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
fsequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Xsequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0>sequential_1_bidirectional_1_backward_lstm_1_while_placeholderosequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
Rsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOp]sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
Esequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/MatMulMatMul]sequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0Zsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Tsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOp_sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
Gsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/MatMul_1MatMul@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_3\sequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Bsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/addAddV2Osequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/MatMul:product:0Qsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Ssequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOp^sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
Dsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_1AddV2Fsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add:z:0[sequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Nsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Dsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/splitSplitWsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/split/split_dim:output:0Hsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
Fsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/SigmoidSigmoidMsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
Hsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Sigmoid_1SigmoidMsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
Bsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/mulMulLsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Sigmoid_1:y:0@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_4*
T0*(
_output_shapes
:�����������
Csequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/TanhTanhMsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
Dsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/mul_1MulJsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Sigmoid:y:0Gsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
Dsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_2AddV2Fsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/mul:z:0Hsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
Hsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Sigmoid_2SigmoidMsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
Esequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Tanh_1TanhHsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
Dsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/mul_2MulLsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Sigmoid_2:y:0Isequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
Asequential_1/bidirectional_1/backward_lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
7sequential_1/bidirectional_1/backward_lstm_1/while/TileTile_sequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Jsequential_1/bidirectional_1/backward_lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
;sequential_1/bidirectional_1/backward_lstm_1/while/SelectV2SelectV2@sequential_1/bidirectional_1/backward_lstm_1/while/Tile:output:0Hsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/mul_2:z:0usequential_1_bidirectional_1_backward_lstm_1_while_selectv2_sequential_1_bidirectional_1_backward_lstm_1_zeros_like_0*
T0*(
_output_shapes
:�����������
Csequential_1/bidirectional_1/backward_lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
9sequential_1/bidirectional_1/backward_lstm_1/while/Tile_1Tile_sequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Lsequential_1/bidirectional_1/backward_lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
Csequential_1/bidirectional_1/backward_lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
9sequential_1/bidirectional_1/backward_lstm_1/while/Tile_2Tile_sequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Lsequential_1/bidirectional_1/backward_lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:����������
=sequential_1/bidirectional_1/backward_lstm_1/while/SelectV2_1SelectV2Bsequential_1/bidirectional_1/backward_lstm_1/while/Tile_1:output:0Hsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/mul_2:z:0@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_3*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/backward_lstm_1/while/SelectV2_2SelectV2Bsequential_1/bidirectional_1/backward_lstm_1/while/Tile_2:output:0Hsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_2:z:0@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_4*
T0*(
_output_shapes
:�����������
Wsequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_1>sequential_1_bidirectional_1_backward_lstm_1_while_placeholderDsequential_1/bidirectional_1/backward_lstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���z
8sequential_1/bidirectional_1/backward_lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
6sequential_1/bidirectional_1/backward_lstm_1/while/addAddV2>sequential_1_bidirectional_1_backward_lstm_1_while_placeholderAsequential_1/bidirectional_1/backward_lstm_1/while/add/y:output:0*
T0*
_output_shapes
: |
:sequential_1/bidirectional_1/backward_lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
8sequential_1/bidirectional_1/backward_lstm_1/while/add_1AddV2rsequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_loop_counterCsequential_1/bidirectional_1/backward_lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
;sequential_1/bidirectional_1/backward_lstm_1/while/IdentityIdentity<sequential_1/bidirectional_1/backward_lstm_1/while/add_1:z:08^sequential_1/bidirectional_1/backward_lstm_1/while/NoOp*
T0*
_output_shapes
: �
=sequential_1/bidirectional_1/backward_lstm_1/while/Identity_1Identitycsequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_max8^sequential_1/bidirectional_1/backward_lstm_1/while/NoOp*
T0*
_output_shapes
: �
=sequential_1/bidirectional_1/backward_lstm_1/while/Identity_2Identity:sequential_1/bidirectional_1/backward_lstm_1/while/add:z:08^sequential_1/bidirectional_1/backward_lstm_1/while/NoOp*
T0*
_output_shapes
: �
=sequential_1/bidirectional_1/backward_lstm_1/while/Identity_3Identitygsequential_1/bidirectional_1/backward_lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^sequential_1/bidirectional_1/backward_lstm_1/while/NoOp*
T0*
_output_shapes
: �
=sequential_1/bidirectional_1/backward_lstm_1/while/Identity_4IdentityDsequential_1/bidirectional_1/backward_lstm_1/while/SelectV2:output:08^sequential_1/bidirectional_1/backward_lstm_1/while/NoOp*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/backward_lstm_1/while/Identity_5IdentityFsequential_1/bidirectional_1/backward_lstm_1/while/SelectV2_1:output:08^sequential_1/bidirectional_1/backward_lstm_1/while/NoOp*
T0*(
_output_shapes
:�����������
=sequential_1/bidirectional_1/backward_lstm_1/while/Identity_6IdentityFsequential_1/bidirectional_1/backward_lstm_1/while/SelectV2_2:output:08^sequential_1/bidirectional_1/backward_lstm_1/while/NoOp*
T0*(
_output_shapes
:�����������
7sequential_1/bidirectional_1/backward_lstm_1/while/NoOpNoOpS^sequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOpU^sequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOpT^sequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "�
;sequential_1_bidirectional_1_backward_lstm_1_while_identityDsequential_1/bidirectional_1/backward_lstm_1/while/Identity:output:0"�
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_1Fsequential_1/bidirectional_1/backward_lstm_1/while/Identity_1:output:0"�
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_2Fsequential_1/bidirectional_1/backward_lstm_1/while/Identity_2:output:0"�
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_3Fsequential_1/bidirectional_1/backward_lstm_1/while/Identity_3:output:0"�
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_4Fsequential_1/bidirectional_1/backward_lstm_1/while/Identity_4:output:0"�
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_5Fsequential_1/bidirectional_1/backward_lstm_1/while/Identity_5:output:0"�
=sequential_1_bidirectional_1_backward_lstm_1_while_identity_6Fsequential_1/bidirectional_1/backward_lstm_1/while/Identity_6:output:0"�
\sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource^sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
]sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
[sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource]sequential_1_bidirectional_1_backward_lstm_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
ssequential_1_bidirectional_1_backward_lstm_1_while_selectv2_sequential_1_bidirectional_1_backward_lstm_1_zeros_likeusequential_1_bidirectional_1_backward_lstm_1_while_selectv2_sequential_1_bidirectional_1_backward_lstm_1_zeros_like_0"�
�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_1_tensorlistfromtensor�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"�
�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_tensorlistfromtensor�sequential_1_bidirectional_1_backward_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b: : : : :����������:����������:����������: : : : : :����������2�
Rsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOpRsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast/ReadVariableOp2�
Tsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOpTsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/Cast_1/ReadVariableOp2�
Ssequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOpSsequential_1/bidirectional_1/backward_lstm_1/while/lstm_cell_1/add_1/ReadVariableOp:w s

_output_shapes
: 
Y
_user_specified_nameA?sequential_1/bidirectional_1/backward_lstm_1/while/loop_counter:hd

_output_shapes
: 
J
_user_specified_name20sequential_1/bidirectional_1/backward_lstm_1/Max:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:��

_output_shapes
: 
n
_user_specified_nameVTsequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack/TensorListFromTensor:��

_output_shapes
: 
p
_user_specified_nameXVsequential_1/bidirectional_1/backward_lstm_1/TensorArrayUnstack_1/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:�}
(
_output_shapes
:����������
Q
_user_specified_name97sequential_1/bidirectional_1/backward_lstm_1/zeros_like
�
�
?sequential_1_bidirectional_1_2_forward_lstm_1_1_while_body_2137|
xsequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_loop_counterm
isequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_maxE
Asequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholderG
Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_1G
Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_2G
Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_3G
Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_4�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor_0�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensor_0t
`sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource_0:
��u
bsequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:	@�p
asequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resource_0:	�B
>sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identityD
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_1D
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_2D
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_3D
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_4D
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_5D
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_6�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensorr
^sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource:
��s
`sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resource:	@�n
_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resource:	���Usequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOp�Wsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOp�Vsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOp�
gsequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Ysequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor_0Asequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholderpsequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
isequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
[sequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensor_0Asequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholderrsequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
Usequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOp`sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
Hsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/MatMulMatMul`sequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read/TensorListGetItem:item:0]sequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Wsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpbsequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
Jsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/MatMul_1MatMulCsequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_3_sequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Esequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/addAddV2Rsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/MatMul:product:0Tsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Vsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOpasequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_1AddV2Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add:z:0^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Qsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/splitSplitZsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/split/split_dim:output:0Ksequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split�
Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/SigmoidSigmoidPsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@�
Ksequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Sigmoid_1SigmoidPsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@�
Esequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/mulMulOsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Sigmoid_1:y:0Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_4*
T0*'
_output_shapes
:���������@�
Fsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/TanhTanhPsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@�
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/mul_1MulMsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Sigmoid:y:0Jsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:���������@�
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_2AddV2Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/mul:z:0Ksequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@�
Ksequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Sigmoid_2SigmoidPsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@�
Hsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Tanh_1TanhKsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:���������@�
Gsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/mul_2MulOsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Sigmoid_2:y:0Lsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
Dsequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
:sequential_1/bidirectional_1_2/forward_lstm_1_1/while/TileTilebsequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Msequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
>sequential_1/bidirectional_1_2/forward_lstm_1_1/while/SelectV2SelectV2Csequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile:output:0Ksequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/mul_2:z:0Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_2*
T0*'
_output_shapes
:���������@�
Fsequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
<sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile_1Tilebsequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Osequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
Fsequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
<sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile_2Tilebsequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Osequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:����������
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/SelectV2_1SelectV2Esequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile_1:output:0Ksequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/mul_2:z:0Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_3*
T0*'
_output_shapes
:���������@�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/SelectV2_2SelectV2Esequential_1/bidirectional_1_2/forward_lstm_1_1/while/Tile_2:output:0Ksequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_2:z:0Csequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_4*
T0*'
_output_shapes
:���������@�
`sequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
Zsequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemCsequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholder_1isequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0Gsequential_1/bidirectional_1_2/forward_lstm_1_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���}
;sequential_1/bidirectional_1_2/forward_lstm_1_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
9sequential_1/bidirectional_1_2/forward_lstm_1_1/while/addAddV2Asequential_1_bidirectional_1_2_forward_lstm_1_1_while_placeholderDsequential_1/bidirectional_1_2/forward_lstm_1_1/while/add/y:output:0*
T0*
_output_shapes
: 
=sequential_1/bidirectional_1_2/forward_lstm_1_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
;sequential_1/bidirectional_1_2/forward_lstm_1_1/while/add_1AddV2xsequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_loop_counterFsequential_1/bidirectional_1_2/forward_lstm_1_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
>sequential_1/bidirectional_1_2/forward_lstm_1_1/while/IdentityIdentity?sequential_1/bidirectional_1_2/forward_lstm_1_1/while/add_1:z:0;^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/NoOp*
T0*
_output_shapes
: �
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_1Identityisequential_1_bidirectional_1_2_forward_lstm_1_1_while_sequential_1_bidirectional_1_2_forward_lstm_1_1_max;^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/NoOp*
T0*
_output_shapes
: �
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_2Identity=sequential_1/bidirectional_1_2/forward_lstm_1_1/while/add:z:0;^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/NoOp*
T0*
_output_shapes
: �
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_3Identityjsequential_1/bidirectional_1_2/forward_lstm_1_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0;^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/NoOp*
T0*
_output_shapes
: �
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_4IdentityGsequential_1/bidirectional_1_2/forward_lstm_1_1/while/SelectV2:output:0;^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/NoOp*
T0*'
_output_shapes
:���������@�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_5IdentityIsequential_1/bidirectional_1_2/forward_lstm_1_1/while/SelectV2_1:output:0;^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/NoOp*
T0*'
_output_shapes
:���������@�
@sequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_6IdentityIsequential_1/bidirectional_1_2/forward_lstm_1_1/while/SelectV2_2:output:0;^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/NoOp*
T0*'
_output_shapes
:���������@�
:sequential_1/bidirectional_1_2/forward_lstm_1_1/while/NoOpNoOpV^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOpX^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOpW^sequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "�
>sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identityGsequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity:output:0"�
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_1Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_1:output:0"�
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_2Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_2:output:0"�
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_3Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_3:output:0"�
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_4Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_4:output:0"�
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_5Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_5:output:0"�
@sequential_1_bidirectional_1_2_forward_lstm_1_1_while_identity_6Isequential_1/bidirectional_1_2/forward_lstm_1_1/while/Identity_6:output:0"�
_sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resourceasequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
`sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resourcebsequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
^sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource`sequential_1_bidirectional_1_2_forward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensor�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensor_0"�
�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor�sequential_1_bidirectional_1_2_forward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_forward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K: : : : :���������@:���������@:���������@: : : : : 2�
Usequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOpUsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOp2�
Wsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOpWsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOp2�
Vsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOpVsequential_1/bidirectional_1_2/forward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOp:z v

_output_shapes
: 
\
_user_specified_nameDBsequential_1/bidirectional_1_2/forward_lstm_1_1/while/loop_counter:kg

_output_shapes
: 
M
_user_specified_name53sequential_1/bidirectional_1_2/forward_lstm_1_1/Max:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:��

_output_shapes
: 
q
_user_specified_nameYWsequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack/TensorListFromTensor:��

_output_shapes
: 
s
_user_specified_name[Ysequential_1/bidirectional_1_2/forward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
@sequential_1_bidirectional_1_2_backward_lstm_1_1_while_body_2368~
zsequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_loop_countero
ksequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_maxF
Bsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholderH
Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_1H
Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_2H
Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_3H
Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_4�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor_0�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensor_0u
asequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource_0:
��v
csequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:	@�q
bsequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resource_0:	�C
?sequential_1_bidirectional_1_2_backward_lstm_1_1_while_identityE
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_1E
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_2E
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_3E
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_4E
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_5E
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_6�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensors
_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource:
��t
asequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resource:	@�o
`sequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resource:	���Vsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOp�Xsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOp�Wsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOp�
hsequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Zsequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor_0Bsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholderqsequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
\sequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensor_0Bsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholderssequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
Vsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOpasequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
Isequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/MatMulMatMulasequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read/TensorListGetItem:item:0^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Xsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpcsequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
Ksequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/MatMul_1MatMulDsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_3`sequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/addAddV2Ssequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/MatMul:product:0Usequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Wsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOpbsequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_1AddV2Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add:z:0_sequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Rsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/splitSplit[sequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/split/split_dim:output:0Lsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split�
Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/SigmoidSigmoidQsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@�
Lsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Sigmoid_1SigmoidQsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@�
Fsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/mulMulPsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Sigmoid_1:y:0Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_4*
T0*'
_output_shapes
:���������@�
Gsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/TanhTanhQsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@�
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/mul_1MulNsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Sigmoid:y:0Ksequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:���������@�
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_2AddV2Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/mul:z:0Lsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@�
Lsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Sigmoid_2SigmoidQsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@�
Isequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Tanh_1TanhLsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:���������@�
Hsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/mul_2MulPsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Sigmoid_2:y:0Msequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
Esequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
;sequential_1/bidirectional_1_2/backward_lstm_1_1/while/TileTilecsequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Nsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
?sequential_1/bidirectional_1_2/backward_lstm_1_1/while/SelectV2SelectV2Dsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile:output:0Lsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/mul_2:z:0Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_2*
T0*'
_output_shapes
:���������@�
Gsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
=sequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile_1Tilecsequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Psequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
Gsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
=sequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile_2Tilecsequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Psequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:����������
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/SelectV2_1SelectV2Fsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile_1:output:0Lsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/mul_2:z:0Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_3*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/SelectV2_2SelectV2Fsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Tile_2:output:0Lsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_2:z:0Dsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_4*
T0*'
_output_shapes
:���������@�
asequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
[sequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemDsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholder_1jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0Hsequential_1/bidirectional_1_2/backward_lstm_1_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���~
<sequential_1/bidirectional_1_2/backward_lstm_1_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
:sequential_1/bidirectional_1_2/backward_lstm_1_1/while/addAddV2Bsequential_1_bidirectional_1_2_backward_lstm_1_1_while_placeholderEsequential_1/bidirectional_1_2/backward_lstm_1_1/while/add/y:output:0*
T0*
_output_shapes
: �
>sequential_1/bidirectional_1_2/backward_lstm_1_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
<sequential_1/bidirectional_1_2/backward_lstm_1_1/while/add_1AddV2zsequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_loop_counterGsequential_1/bidirectional_1_2/backward_lstm_1_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
?sequential_1/bidirectional_1_2/backward_lstm_1_1/while/IdentityIdentity@sequential_1/bidirectional_1_2/backward_lstm_1_1/while/add_1:z:0<^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/NoOp*
T0*
_output_shapes
: �
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_1Identityksequential_1_bidirectional_1_2_backward_lstm_1_1_while_sequential_1_bidirectional_1_2_backward_lstm_1_1_max<^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/NoOp*
T0*
_output_shapes
: �
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_2Identity>sequential_1/bidirectional_1_2/backward_lstm_1_1/while/add:z:0<^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/NoOp*
T0*
_output_shapes
: �
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_3Identityksequential_1/bidirectional_1_2/backward_lstm_1_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/NoOp*
T0*
_output_shapes
: �
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_4IdentityHsequential_1/bidirectional_1_2/backward_lstm_1_1/while/SelectV2:output:0<^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/NoOp*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_5IdentityJsequential_1/bidirectional_1_2/backward_lstm_1_1/while/SelectV2_1:output:0<^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/NoOp*
T0*'
_output_shapes
:���������@�
Asequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_6IdentityJsequential_1/bidirectional_1_2/backward_lstm_1_1/while/SelectV2_2:output:0<^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/NoOp*
T0*'
_output_shapes
:���������@�
;sequential_1/bidirectional_1_2/backward_lstm_1_1/while/NoOpNoOpW^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOpY^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOpX^sequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "�
?sequential_1_bidirectional_1_2_backward_lstm_1_1_while_identityHsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity:output:0"�
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_1Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_1:output:0"�
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_2Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_2:output:0"�
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_3Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_3:output:0"�
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_4Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_4:output:0"�
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_5Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_5:output:0"�
Asequential_1_bidirectional_1_2_backward_lstm_1_1_while_identity_6Jsequential_1/bidirectional_1_2/backward_lstm_1_1/while/Identity_6:output:0"�
`sequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resourcebsequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
asequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resourcecsequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
_sequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resourceasequential_1_bidirectional_1_2_backward_lstm_1_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensor�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_1_tensorlistfromtensor_0"�
�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor�sequential_1_bidirectional_1_2_backward_lstm_1_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_2_backward_lstm_1_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K: : : : :���������@:���������@:���������@: : : : : 2�
Vsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOpVsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast/ReadVariableOp2�
Xsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOpXsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/Cast_1/ReadVariableOp2�
Wsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOpWsequential_1/bidirectional_1_2/backward_lstm_1_1/while/lstm_cell_1/add_1/ReadVariableOp:{ w

_output_shapes
: 
]
_user_specified_nameECsequential_1/bidirectional_1_2/backward_lstm_1_1/while/loop_counter:lh

_output_shapes
: 
N
_user_specified_name64sequential_1/bidirectional_1_2/backward_lstm_1_1/Max:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:��

_output_shapes
: 
r
_user_specified_nameZXsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack/TensorListFromTensor:��

_output_shapes
: 
t
_user_specified_name\Zsequential_1/bidirectional_1_2/backward_lstm_1_1/TensorArrayUnstack_1/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
<sequential_1_bidirectional_1_backward_lstm_1_while_cond_1887v
rsequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_loop_counterg
csequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_maxB
>sequential_1_bidirectional_1_backward_lstm_1_while_placeholderD
@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_1D
@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_2D
@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_3D
@sequential_1_bidirectional_1_backward_lstm_1_while_placeholder_4�
�sequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_cond_1887___redundant_placeholder0�
�sequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_cond_1887___redundant_placeholder1�
�sequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_cond_1887___redundant_placeholder2�
�sequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_cond_1887___redundant_placeholder3�
�sequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_cond_1887___redundant_placeholder4�
�sequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_cond_1887___redundant_placeholder5?
;sequential_1_bidirectional_1_backward_lstm_1_while_identity
{
9sequential_1/bidirectional_1/backward_lstm_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :x�
7sequential_1/bidirectional_1/backward_lstm_1/while/LessLess>sequential_1_bidirectional_1_backward_lstm_1_while_placeholderBsequential_1/bidirectional_1/backward_lstm_1/while/Less/y:output:0*
T0*
_output_shapes
: �
9sequential_1/bidirectional_1/backward_lstm_1/while/Less_1Lessrsequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_while_loop_countercsequential_1_bidirectional_1_backward_lstm_1_while_sequential_1_bidirectional_1_backward_lstm_1_max*
T0*
_output_shapes
: �
=sequential_1/bidirectional_1/backward_lstm_1/while/LogicalAnd
LogicalAnd=sequential_1/bidirectional_1/backward_lstm_1/while/Less_1:z:0;sequential_1/bidirectional_1/backward_lstm_1/while/Less:z:0*
_output_shapes
: �
;sequential_1/bidirectional_1/backward_lstm_1/while/IdentityIdentityAsequential_1/bidirectional_1/backward_lstm_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "�
;sequential_1_bidirectional_1_backward_lstm_1_while_identityDsequential_1/bidirectional_1/backward_lstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\: : : : :����������:����������:����������:::::::w s

_output_shapes
: 
Y
_user_specified_nameA?sequential_1/bidirectional_1/backward_lstm_1/while/loop_counter:hd

_output_shapes
: 
J
_user_specified_name20sequential_1/bidirectional_1/backward_lstm_1/Max:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
::

_output_shapes
::

_output_shapes
:
�
�
+__inference_signature_wrapper___call___2628
input_layer
unknown:���
	unknown_0:	
	unknown_1:
��
	unknown_2:
��
	unknown_3:	�
	unknown_4:	
	unknown_5:
��
	unknown_6:
��
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	

unknown_13:
��

unknown_14:	@�

unknown_15:	�

unknown_16:	

unknown_17:
��

unknown_18:	@�

unknown_19:	�

unknown_20:	�@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *"
fR
__inference___call___2501o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:���������x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������x
%
_user_specified_nameinput_layer:$ 

_user_specified_name2568:$ 

_user_specified_name2570:$ 

_user_specified_name2572:$ 

_user_specified_name2574:$ 

_user_specified_name2576:$ 

_user_specified_name2578:$ 

_user_specified_name2580:$ 

_user_specified_name2582:$	 

_user_specified_name2584:$
 

_user_specified_name2586:$ 

_user_specified_name2588:$ 

_user_specified_name2590:$ 

_user_specified_name2592:$ 

_user_specified_name2594:$ 

_user_specified_name2596:$ 

_user_specified_name2598:$ 

_user_specified_name2600:$ 

_user_specified_name2602:$ 

_user_specified_name2604:$ 

_user_specified_name2606:$ 

_user_specified_name2608:$ 

_user_specified_name2610:$ 

_user_specified_name2612:$ 

_user_specified_name2614:$ 

_user_specified_name2616:$ 

_user_specified_name2618:$ 

_user_specified_name2620:$ 

_user_specified_name2622:$ 

_user_specified_name2624"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
9
input_layer*
serve_input_layer:0���������x<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict*�
serving_default�
C
input_layer4
serving_default_input_layer:0���������x>
output_02
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:�,
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30"
trackable_list_wrapper
�
0

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
%19
&20"
trackable_list_wrapper
f
	0
1
2
3
4
5
6
"7
#8
$9"
trackable_list_wrapper
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
@25
A26
B27
C28"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Dtrace_02�
__inference___call___2501�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_layer���������xzDtrace_0
7
	Eserve
Fserving_default"
signature_map
4:2���2sequential/embedding/embeddings
/:-	2#seed_generator/seed_generator_state
1:/
��2forward_lstm/lstm_cell/kernel
;:9
��2'forward_lstm/lstm_cell/recurrent_kernel
*:(�2forward_lstm/lstm_cell/bias
1:/	2%seed_generator_3/seed_generator_state
2:0
��2backward_lstm/lstm_cell/kernel
<::
��2(backward_lstm/lstm_cell/recurrent_kernel
+:)�2backward_lstm/lstm_cell/bias
1:/	2%seed_generator_2/seed_generator_state
3:1�2$sequential/batch_normalization/gamma
2:0�2#sequential/batch_normalization/beta
7:5�2*sequential/batch_normalization/moving_mean
;:9�2.sequential/batch_normalization/moving_variance
3:1
��2forward_lstm_1/lstm_cell/kernel
<::	@�2)forward_lstm_1/lstm_cell/recurrent_kernel
,:*�2forward_lstm_1/lstm_cell/bias
1:/	2%seed_generator_6/seed_generator_state
4:2
��2 backward_lstm_1/lstm_cell/kernel
=:;	@�2*backward_lstm_1/lstm_cell/recurrent_kernel
-:+�2backward_lstm_1/lstm_cell/bias
1:/	2%seed_generator_5/seed_generator_state
*:(	�@2sequential/dense/kernel
#:!@2sequential/dense/bias
4:2@2&sequential/batch_normalization_1/gamma
3:1@2%sequential/batch_normalization_1/beta
8:6@2,sequential/batch_normalization_1/moving_mean
<::@20sequential/batch_normalization_1/moving_variance
1:/	2%seed_generator_7/seed_generator_state
+:)@2sequential/dense_1/kernel
%:#2sequential/dense_1/bias
*:(�2forward_lstm/lstm_cell/bias
3:1�2$sequential/batch_normalization/gamma
4:2@2&sequential/batch_normalization_1/gamma
4:2���2sequential/embedding/embeddings
;:9
��2'forward_lstm/lstm_cell/recurrent_kernel
4:2
��2 backward_lstm_1/lstm_cell/kernel
+:)�2backward_lstm/lstm_cell/bias
,:*�2forward_lstm_1/lstm_cell/bias
<::
��2(backward_lstm/lstm_cell/recurrent_kernel
3:1@2%sequential/batch_normalization_1/beta
%:#2sequential/dense_1/bias
<::	@�2)forward_lstm_1/lstm_cell/recurrent_kernel
+:)@2sequential/dense_1/kernel
1:/
��2forward_lstm/lstm_cell/kernel
=:;	@�2*backward_lstm_1/lstm_cell/recurrent_kernel
3:1
��2forward_lstm_1/lstm_cell/kernel
-:+�2backward_lstm_1/lstm_cell/bias
#:!@2sequential/dense/bias
2:0
��2backward_lstm/lstm_cell/kernel
2:0�2#sequential/batch_normalization/beta
*:(	�@2sequential/dense/kernel
1:/	2%seed_generator_2/seed_generator_state
7:5�2*sequential/batch_normalization/moving_mean
<::@20sequential/batch_normalization_1/moving_variance
1:/	2%seed_generator_6/seed_generator_state
1:/	2%seed_generator_5/seed_generator_state
;:9�2.sequential/batch_normalization/moving_variance
8:6@2,sequential/batch_normalization_1/moving_mean
1:/	2%seed_generator_3/seed_generator_state
�B�
__inference___call___2501input_layer"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_signature_wrapper___call___2565input_layer"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
  

kwonlyargs�
jinput_layer
kwonlydefaults
 
annotations� *
 
�B�
+__inference_signature_wrapper___call___2628input_layer"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
  

kwonlyargs�
jinput_layer
kwonlydefaults
 
annotations� *
 �
__inference___call___2501x
"# !%&4�1
*�'
%�"
input_layer���������x
� "!�
unknown����������
+__inference_signature_wrapper___call___2565�
"# !%&C�@
� 
9�6
4
input_layer%�"
input_layer���������x"3�0
.
output_0"�
output_0����������
+__inference_signature_wrapper___call___2628�
"# !%&C�@
� 
9�6
4
input_layer%�"
input_layer���������x"3�0
.
output_0"�
output_0���������