ľ
´%%
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
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
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
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
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
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
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
#
	LogicalOr
x

y

z

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12b'v1.13.1-0-g6612da8951'ŁŤ
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙'*
shape:˙˙˙˙˙˙˙˙˙'
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_2Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
Ľ
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"'      *"
_class
loc:@pi/dense/kernel

.pi/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *bśŐ˝*"
_class
loc:@pi/dense/kernel

.pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *bśŐ=*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: 
ď
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	'*

seed *
T0*"
_class
loc:@pi/dense/kernel*
seed2
Ú
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
: 
í
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	'
ß
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	'
Š
pi/dense/kernel
VariableV2*
	container *
shape:	'*
dtype0*
_output_shapes
:	'*
shared_name *"
_class
loc:@pi/dense/kernel
Ô
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel

pi/dense/kernel/readIdentitypi/dense/kernel*
_output_shapes
:	'*
T0*"
_class
loc:@pi/dense/kernel

pi/dense/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@pi/dense/bias*
dtype0*
_output_shapes	
:

pi/dense/bias
VariableV2*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ż
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
u
pi/dense/bias/readIdentitypi/dense/bias*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:

pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
pi/dense/TanhTanhpi/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
:

0pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *óľ˝*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
: 

0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *óľ=*$
_class
loc:@pi/dense_1/kernel
ö
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@pi/dense_1/kernel*
seed2*
dtype0* 
_output_shapes
:
*

seed 
â
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@pi/dense_1/kernel
ö
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

č
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:

Ż
pi/dense_1/kernel
VariableV2*
shared_name *$
_class
loc:@pi/dense_1/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel

pi/dense_1/kernel/readIdentitypi/dense_1/kernel* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel

!pi/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *"
_class
loc:@pi/dense_1/bias
Ą
pi/dense_1/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@pi/dense_1/bias
Ç
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
{
pi/dense_1/bias/readIdentitypi/dense_1/bias*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:

pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *$
_class
loc:@pi/dense_2/kernel

0pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ş7ž*$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
: 

0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ş7>*$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
: 
ő
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@pi/dense_2/kernel*
seed2**
dtype0*
_output_shapes
:	*

seed 
â
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: 
ő
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
ç
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
­
pi/dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container *
shape:	
Ü
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel

pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes
:	*
T0*$
_class
loc:@pi/dense_2/kernel

!pi/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@pi/dense_2/bias*
dtype0*
_output_shapes
:

pi/dense_2/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@pi/dense_2/bias
Ć
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

pi/log_std/initial_valueConst*E
value<B:"0   ż   ż   ż   ż   ż   ż   ż   ż   ż   ż   ż   ż*
dtype0*
_output_shapes
:
v

pi/log_std
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
Ž
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
k
pi/log_std/readIdentity
pi/log_std*
_output_shapes
:*
T0*
_class
loc:@pi/log_std
C
pi/ExpExppi/log_std/read*
T0*
_output_shapes
:
Z
pi/ShapeShapepi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
Z
pi/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
pi/random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*
T0*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
seed2?*

seed 

pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
pi/mulMulpi/random_normalpi/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
pi/addAddpi/dense_2/BiasAddpi/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
pi/subSubPlaceholder_1pi/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
E
pi/Exp_1Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
J
pi/add_1Addpi/Exp_1
pi/add_1/y*
T0*
_output_shapes
:
Y

pi/truedivRealDivpi/subpi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
pi/pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
U
pi/powPow
pi/truedivpi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_1/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
pi/mul_1Mul
pi/mul_1/xpi/log_std/read*
T0*
_output_shapes
:
S
pi/add_2Addpi/powpi/mul_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *?ë?
W
pi/add_3Addpi/add_2
pi/add_3/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_2/xConst*
valueB
 *   ż*
dtype0*
_output_shapes
: 
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Z
pi/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
|
pi/SumSumpi/mul_2pi/Sum/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0
]
pi/sub_1Subpi/addpi/dense_2/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
E
pi/Exp_2Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_4/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
J
pi/add_4Addpi/Exp_2
pi/add_4/y*
T0*
_output_shapes
:
]
pi/truediv_1RealDivpi/sub_1pi/add_4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
[
pi/pow_1Powpi/truediv_1
pi/pow_1/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_3/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
pi/mul_3Mul
pi/mul_3/xpi/log_std/read*
T0*
_output_shapes
:
U
pi/add_5Addpi/pow_1pi/mul_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O

pi/add_6/yConst*
valueB
 *?ë?*
dtype0*
_output_shapes
: 
W
pi/add_6Addpi/add_5
pi/add_6/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O

pi/mul_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ż
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
pi/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 

pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/v/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"'      *!
_class
loc:@v/dense/kernel*
dtype0*
_output_shapes
:

-v/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *bśŐ˝*!
_class
loc:@v/dense/kernel

-v/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *bśŐ=*!
_class
loc:@v/dense/kernel
ě
7v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform/v/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*!
_class
loc:@v/dense/kernel*
seed2g*
dtype0*
_output_shapes
:	'
Ö
-v/dense/kernel/Initializer/random_uniform/subSub-v/dense/kernel/Initializer/random_uniform/max-v/dense/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
: 
é
-v/dense/kernel/Initializer/random_uniform/mulMul7v/dense/kernel/Initializer/random_uniform/RandomUniform-v/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	'*
T0*!
_class
loc:@v/dense/kernel
Ű
)v/dense/kernel/Initializer/random_uniformAdd-v/dense/kernel/Initializer/random_uniform/mul-v/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	'*
T0*!
_class
loc:@v/dense/kernel
§
v/dense/kernel
VariableV2*
shape:	'*
dtype0*
_output_shapes
:	'*
shared_name *!
_class
loc:@v/dense/kernel*
	container 
Đ
v/dense/kernel/AssignAssignv/dense/kernel)v/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
|
v/dense/kernel/readIdentityv/dense/kernel*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	'

v/dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@v/dense/bias*
dtype0*
_output_shapes	
:

v/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@v/dense/bias
ť
v/dense/bias/AssignAssignv/dense/biasv/dense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
r
v/dense/bias/readIdentityv/dense/bias*
_output_shapes	
:*
T0*
_class
loc:@v/dense/bias

v/dense/MatMulMatMulPlaceholderv/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

v/dense/BiasAddBiasAddv/dense/MatMulv/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
v/dense/TanhTanhv/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
1v/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *#
_class
loc:@v/dense_1/kernel*
dtype0*
_output_shapes
:

/v/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *óľ˝*#
_class
loc:@v/dense_1/kernel

/v/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *óľ=*#
_class
loc:@v/dense_1/kernel*
dtype0*
_output_shapes
: 
ó
9v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@v/dense_1/kernel*
seed2x*
dtype0* 
_output_shapes
:

Ţ
/v/dense_1/kernel/Initializer/random_uniform/subSub/v/dense_1/kernel/Initializer/random_uniform/max/v/dense_1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes
: 
ň
/v/dense_1/kernel/Initializer/random_uniform/mulMul9v/dense_1/kernel/Initializer/random_uniform/RandomUniform/v/dense_1/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@v/dense_1/kernel* 
_output_shapes
:

ä
+v/dense_1/kernel/Initializer/random_uniformAdd/v/dense_1/kernel/Initializer/random_uniform/mul/v/dense_1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_1/kernel* 
_output_shapes
:

­
v/dense_1/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *#
_class
loc:@v/dense_1/kernel
Ů
v/dense_1/kernel/AssignAssignv/dense_1/kernel+v/dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel

v/dense_1/kernel/readIdentityv/dense_1/kernel*
T0*#
_class
loc:@v/dense_1/kernel* 
_output_shapes
:


 v/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *!
_class
loc:@v/dense_1/bias

v/dense_1/bias
VariableV2*!
_class
loc:@v/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ă
v/dense_1/bias/AssignAssignv/dense_1/bias v/dense_1/bias/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
x
v/dense_1/bias/readIdentityv/dense_1/bias*
_output_shapes	
:*
T0*!
_class
loc:@v/dense_1/bias

v/dense_1/MatMulMatMulv/dense/Tanhv/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

v/dense_1/BiasAddBiasAddv/dense_1/MatMulv/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
v/dense_1/TanhTanhv/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
1v/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *#
_class
loc:@v/dense_2/kernel*
dtype0*
_output_shapes
:

/v/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *Ivž*#
_class
loc:@v/dense_2/kernel*
dtype0*
_output_shapes
: 

/v/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*#
_class
loc:@v/dense_2/kernel*
dtype0*
_output_shapes
: 
ó
9v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_2/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@v/dense_2/kernel*
seed2*
dtype0*
_output_shapes
:	*

seed 
Ţ
/v/dense_2/kernel/Initializer/random_uniform/subSub/v/dense_2/kernel/Initializer/random_uniform/max/v/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@v/dense_2/kernel
ń
/v/dense_2/kernel/Initializer/random_uniform/mulMul9v/dense_2/kernel/Initializer/random_uniform/RandomUniform/v/dense_2/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes
:	
ă
+v/dense_2/kernel/Initializer/random_uniformAdd/v/dense_2/kernel/Initializer/random_uniform/mul/v/dense_2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes
:	
Ť
v/dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *#
_class
loc:@v/dense_2/kernel*
	container *
shape:	
Ř
v/dense_2/kernel/AssignAssignv/dense_2/kernel+v/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel

v/dense_2/kernel/readIdentityv/dense_2/kernel*
_output_shapes
:	*
T0*#
_class
loc:@v/dense_2/kernel

 v/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *!
_class
loc:@v/dense_2/bias

v/dense_2/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@v/dense_2/bias*
	container 
Â
v/dense_2/bias/AssignAssignv/dense_2/bias v/dense_2/bias/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
w
v/dense_2/bias/readIdentityv/dense_2/bias*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:

v/dense_2/MatMulMatMulv/dense_1/Tanhv/dense_2/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

v/dense_2/BiasAddBiasAddv/dense_2/MatMulv/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
	v/SqueezeSqueezev/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
O
subSubpi/SumPlaceholder_4*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
=
ExpExpsub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
Z
GreaterGreaterPlaceholder_2	Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *?
N
mulMulmul/xPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mul_1/xConst*
valueB
 *ÍĚL?*
dtype0*
_output_shapes
: 
R
mul_1Mulmul_1/xPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
S
SelectSelectGreatermulmul_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
N
mul_2MulExpPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
MinimumMinimummul_2Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
Z
MeanMeanMinimumConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
1
NegNegMean*
T0*
_output_shapes
: 
T
sub_1SubPlaceholder_3	v/Squeeze*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
F
powPowsub_1pow/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Z
Mean_1MeanpowConst_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Q
sub_2SubPlaceholder_4pi/Sum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_2Meansub_2Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
B
Neg_1Negpi/Sum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_3MeanNeg_1Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
P
Greater_1/yConst*
valueB
 *?*
dtype0*
_output_shapes
: 
T
	Greater_1GreaterExpGreater_1/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
Less/yConst*
valueB
 *ÍĚL?*
dtype0*
_output_shapes
: 
G
LessLessExpLess/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
	LogicalOr	LogicalOr	Greater_1Less*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
CastCast	LogicalOr*

SrcT0
*
Truncate( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_4MeanCastConst_4*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients/Mean_grad/ShapeShapeMinimum*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/Mean_grad/Shape_1ShapeMinimum*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients/Minimum_grad/ShapeShapemul_2*
_output_shapes
:*
T0*
out_type0
d
gradients/Minimum_grad/Shape_1ShapeSelect*
T0*
out_type0*
_output_shapes
:
y
gradients/Minimum_grad/Shape_2Shapegradients/Mean_grad/truediv*
T0*
out_type0*
_output_shapes
:
g
"gradients/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¨
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
˛
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_grad/truedivgradients/Minimum_grad/zeros*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_grad/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ľ
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
ć
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*1
_class'
%#loc:@gradients/Minimum_grad/Reshape
ě
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1
]
gradients/mul_2_grad/ShapeShapeExp*
T0*
out_type0*
_output_shapes
:
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
ş
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
Ţ
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
ä
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Exp_grad/mulMul-gradients/mul_2_grad/tuple/control_dependencyExp*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
gradients/sub_grad/ShapeShapepi/Sum*
_output_shapes
:*
T0*
out_type0
g
gradients/sub_grad/Shape_1ShapePlaceholder_4*
_output_shapes
:*
T0*
out_type0
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ö
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
Ü
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients/pi/Sum_grad/ShapeShapepi/mul_2*
T0*
out_type0*
_output_shapes
:

gradients/pi/Sum_grad/SizeConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
§
gradients/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
­
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape

gradients/pi/Sum_grad/Shape_1Const*
valueB *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 

!gradients/pi/Sum_grad/range/startConst*
value	B : *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 

!gradients/pi/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
Ţ
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*

Tidx0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:

 gradients/pi/Sum_grad/Fill/valueConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ć
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*
T0*

index_type0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 

#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
N*
_output_shapes
:

gradients/pi/Sum_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ă
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
ť
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
Ă
gradients/pi/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency#gradients/pi/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ľ
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
`
gradients/pi/mul_2_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
g
gradients/pi/mul_2_grad/Shape_1Shapepi/add_3*
T0*
out_type0*
_output_shapes
:
Ă
-gradients/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_2_grad/Shapegradients/pi/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
z
gradients/pi/mul_2_grad/MulMulgradients/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients/pi/mul_2_grad/SumSumgradients/pi/mul_2_grad/Mul-gradients/pi/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/pi/mul_2_grad/ReshapeReshapegradients/pi/mul_2_grad/Sumgradients/pi/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
~
gradients/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients/pi/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients/pi/mul_2_grad/Sum_1Sumgradients/pi/mul_2_grad/Mul_1/gradients/pi/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ź
!gradients/pi/mul_2_grad/Reshape_1Reshapegradients/pi/mul_2_grad/Sum_1gradients/pi/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
(gradients/pi/mul_2_grad/tuple/group_depsNoOp ^gradients/pi/mul_2_grad/Reshape"^gradients/pi/mul_2_grad/Reshape_1
Ý
0gradients/pi/mul_2_grad/tuple/control_dependencyIdentitygradients/pi/mul_2_grad/Reshape)^gradients/pi/mul_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/mul_2_grad/Reshape*
_output_shapes
: 
ô
2gradients/pi/mul_2_grad/tuple/control_dependency_1Identity!gradients/pi/mul_2_grad/Reshape_1)^gradients/pi/mul_2_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*4
_class*
(&loc:@gradients/pi/mul_2_grad/Reshape_1
e
gradients/pi/add_3_grad/ShapeShapepi/add_2*
T0*
out_type0*
_output_shapes
:
b
gradients/pi/add_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ă
-gradients/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_3_grad/Shapegradients/pi/add_3_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ĺ
gradients/pi/add_3_grad/SumSum2gradients/pi/mul_2_grad/tuple/control_dependency_1-gradients/pi/add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ś
gradients/pi/add_3_grad/ReshapeReshapegradients/pi/add_3_grad/Sumgradients/pi/add_3_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
É
gradients/pi/add_3_grad/Sum_1Sum2gradients/pi/mul_2_grad/tuple/control_dependency_1/gradients/pi/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

!gradients/pi/add_3_grad/Reshape_1Reshapegradients/pi/add_3_grad/Sum_1gradients/pi/add_3_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
v
(gradients/pi/add_3_grad/tuple/group_depsNoOp ^gradients/pi/add_3_grad/Reshape"^gradients/pi/add_3_grad/Reshape_1
î
0gradients/pi/add_3_grad/tuple/control_dependencyIdentitygradients/pi/add_3_grad/Reshape)^gradients/pi/add_3_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
2gradients/pi/add_3_grad/tuple/control_dependency_1Identity!gradients/pi/add_3_grad/Reshape_1)^gradients/pi/add_3_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/pi/add_3_grad/Reshape_1*
_output_shapes
: 
c
gradients/pi/add_2_grad/ShapeShapepi/pow*
T0*
out_type0*
_output_shapes
:
i
gradients/pi/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ă
-gradients/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_2_grad/Shapegradients/pi/add_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ă
gradients/pi/add_2_grad/SumSum0gradients/pi/add_3_grad/tuple/control_dependency-gradients/pi/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ś
gradients/pi/add_2_grad/ReshapeReshapegradients/pi/add_2_grad/Sumgradients/pi/add_2_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ç
gradients/pi/add_2_grad/Sum_1Sum0gradients/pi/add_3_grad/tuple/control_dependency/gradients/pi/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

!gradients/pi/add_2_grad/Reshape_1Reshapegradients/pi/add_2_grad/Sum_1gradients/pi/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
v
(gradients/pi/add_2_grad/tuple/group_depsNoOp ^gradients/pi/add_2_grad/Reshape"^gradients/pi/add_2_grad/Reshape_1
î
0gradients/pi/add_2_grad/tuple/control_dependencyIdentitygradients/pi/add_2_grad/Reshape)^gradients/pi/add_2_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@gradients/pi/add_2_grad/Reshape
ç
2gradients/pi/add_2_grad/tuple/control_dependency_1Identity!gradients/pi/add_2_grad/Reshape_1)^gradients/pi/add_2_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/pi/add_2_grad/Reshape_1*
_output_shapes
:
e
gradients/pi/pow_grad/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
`
gradients/pi/pow_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
˝
+gradients/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/pow_grad/Shapegradients/pi/pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/pi/pow_grad/mulMul0gradients/pi/add_2_grad/tuple/control_dependencypi/pow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients/pi/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
gradients/pi/pow_grad/subSubpi/pow/ygradients/pi/pow_grad/sub/y*
_output_shapes
: *
T0
y
gradients/pi/pow_grad/PowPow
pi/truedivgradients/pi/pow_grad/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/pi/pow_grad/mul_1Mulgradients/pi/pow_grad/mulgradients/pi/pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
gradients/pi/pow_grad/SumSumgradients/pi/pow_grad/mul_1+gradients/pi/pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
 
gradients/pi/pow_grad/ReshapeReshapegradients/pi/pow_grad/Sumgradients/pi/pow_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
gradients/pi/pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/pi/pow_grad/GreaterGreater
pi/truedivgradients/pi/pow_grad/Greater/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
%gradients/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
j
%gradients/pi/pow_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
š
gradients/pi/pow_grad/ones_likeFill%gradients/pi/pow_grad/ones_like/Shape%gradients/pi/pow_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
gradients/pi/pow_grad/SelectSelectgradients/pi/pow_grad/Greater
pi/truedivgradients/pi/pow_grad/ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
gradients/pi/pow_grad/LogLoggradients/pi/pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
 gradients/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
gradients/pi/pow_grad/Select_1Selectgradients/pi/pow_grad/Greatergradients/pi/pow_grad/Log gradients/pi/pow_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/pi/pow_grad/mul_2Mul0gradients/pi/add_2_grad/tuple/control_dependencypi/pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/pi/pow_grad/mul_3Mulgradients/pi/pow_grad/mul_2gradients/pi/pow_grad/Select_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients/pi/pow_grad/Sum_1Sumgradients/pi/pow_grad/mul_3-gradients/pi/pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/pi/pow_grad/Reshape_1Reshapegradients/pi/pow_grad/Sum_1gradients/pi/pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
p
&gradients/pi/pow_grad/tuple/group_depsNoOp^gradients/pi/pow_grad/Reshape ^gradients/pi/pow_grad/Reshape_1
ć
.gradients/pi/pow_grad/tuple/control_dependencyIdentitygradients/pi/pow_grad/Reshape'^gradients/pi/pow_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/pi/pow_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
0gradients/pi/pow_grad/tuple/control_dependency_1Identitygradients/pi/pow_grad/Reshape_1'^gradients/pi/pow_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/pow_grad/Reshape_1*
_output_shapes
: 
`
gradients/pi/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
i
gradients/pi/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ă
-gradients/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_1_grad/Shapegradients/pi/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/pi/mul_1_grad/MulMul2gradients/pi/add_2_grad/tuple/control_dependency_1pi/log_std/read*
T0*
_output_shapes
:
Ź
gradients/pi/mul_1_grad/SumSumgradients/pi/mul_1_grad/Mul-gradients/pi/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

gradients/pi/mul_1_grad/ReshapeReshapegradients/pi/mul_1_grad/Sumgradients/pi/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

gradients/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x2gradients/pi/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
ś
gradients/pi/mul_1_grad/Sum_1Sumgradients/pi/mul_1_grad/Mul_1/gradients/pi/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

!gradients/pi/mul_1_grad/Reshape_1Reshapegradients/pi/mul_1_grad/Sum_1gradients/pi/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
v
(gradients/pi/mul_1_grad/tuple/group_depsNoOp ^gradients/pi/mul_1_grad/Reshape"^gradients/pi/mul_1_grad/Reshape_1
Ý
0gradients/pi/mul_1_grad/tuple/control_dependencyIdentitygradients/pi/mul_1_grad/Reshape)^gradients/pi/mul_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/mul_1_grad/Reshape*
_output_shapes
: 
ç
2gradients/pi/mul_1_grad/tuple/control_dependency_1Identity!gradients/pi/mul_1_grad/Reshape_1)^gradients/pi/mul_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/pi/mul_1_grad/Reshape_1*
_output_shapes
:
e
gradients/pi/truediv_grad/ShapeShapepi/sub*
_output_shapes
:*
T0*
out_type0
k
!gradients/pi/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
É
/gradients/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/truediv_grad/Shape!gradients/pi/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

!gradients/pi/truediv_grad/RealDivRealDiv.gradients/pi/pow_grad/tuple/control_dependencypi/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients/pi/truediv_grad/SumSum!gradients/pi/truediv_grad/RealDiv/gradients/pi/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ź
!gradients/pi/truediv_grad/ReshapeReshapegradients/pi/truediv_grad/Sumgradients/pi/truediv_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
^
gradients/pi/truediv_grad/NegNegpi/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

#gradients/pi/truediv_grad/RealDiv_1RealDivgradients/pi/truediv_grad/Negpi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

#gradients/pi/truediv_grad/RealDiv_2RealDiv#gradients/pi/truediv_grad/RealDiv_1pi/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients/pi/truediv_grad/mulMul.gradients/pi/pow_grad/tuple/control_dependency#gradients/pi/truediv_grad/RealDiv_2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients/pi/truediv_grad/Sum_1Sumgradients/pi/truediv_grad/mul1gradients/pi/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ľ
#gradients/pi/truediv_grad/Reshape_1Reshapegradients/pi/truediv_grad/Sum_1!gradients/pi/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
|
*gradients/pi/truediv_grad/tuple/group_depsNoOp"^gradients/pi/truediv_grad/Reshape$^gradients/pi/truediv_grad/Reshape_1
ö
2gradients/pi/truediv_grad/tuple/control_dependencyIdentity!gradients/pi/truediv_grad/Reshape+^gradients/pi/truediv_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*4
_class*
(&loc:@gradients/pi/truediv_grad/Reshape
ď
4gradients/pi/truediv_grad/tuple/control_dependency_1Identity#gradients/pi/truediv_grad/Reshape_1+^gradients/pi/truediv_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/pi/truediv_grad/Reshape_1*
_output_shapes
:
h
gradients/pi/sub_grad/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
o
gradients/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
˝
+gradients/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/sub_grad/Shapegradients/pi/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Á
gradients/pi/sub_grad/SumSum2gradients/pi/truediv_grad/tuple/control_dependency+gradients/pi/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
 
gradients/pi/sub_grad/ReshapeReshapegradients/pi/sub_grad/Sumgradients/pi/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
gradients/pi/sub_grad/Sum_1Sum2gradients/pi/truediv_grad/tuple/control_dependency-gradients/pi/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
`
gradients/pi/sub_grad/NegNeggradients/pi/sub_grad/Sum_1*
T0*
_output_shapes
:
¤
gradients/pi/sub_grad/Reshape_1Reshapegradients/pi/sub_grad/Neggradients/pi/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
&gradients/pi/sub_grad/tuple/group_depsNoOp^gradients/pi/sub_grad/Reshape ^gradients/pi/sub_grad/Reshape_1
ć
.gradients/pi/sub_grad/tuple/control_dependencyIdentitygradients/pi/sub_grad/Reshape'^gradients/pi/sub_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/pi/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
0gradients/pi/sub_grad/tuple/control_dependency_1Identitygradients/pi/sub_grad/Reshape_1'^gradients/pi/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@gradients/pi/sub_grad/Reshape_1
g
gradients/pi/add_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
b
gradients/pi/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ă
-gradients/pi/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_1_grad/Shapegradients/pi/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
gradients/pi/add_1_grad/SumSum4gradients/pi/truediv_grad/tuple/control_dependency_1-gradients/pi/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/pi/add_1_grad/ReshapeReshapegradients/pi/add_1_grad/Sumgradients/pi/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
É
gradients/pi/add_1_grad/Sum_1Sum4gradients/pi/truediv_grad/tuple/control_dependency_1/gradients/pi/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

!gradients/pi/add_1_grad/Reshape_1Reshapegradients/pi/add_1_grad/Sum_1gradients/pi/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
v
(gradients/pi/add_1_grad/tuple/group_depsNoOp ^gradients/pi/add_1_grad/Reshape"^gradients/pi/add_1_grad/Reshape_1
á
0gradients/pi/add_1_grad/tuple/control_dependencyIdentitygradients/pi/add_1_grad/Reshape)^gradients/pi/add_1_grad/tuple/group_deps*
_output_shapes
:*
T0*2
_class(
&$loc:@gradients/pi/add_1_grad/Reshape
ă
2gradients/pi/add_1_grad/tuple/control_dependency_1Identity!gradients/pi/add_1_grad/Reshape_1)^gradients/pi/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*4
_class*
(&loc:@gradients/pi/add_1_grad/Reshape_1
Ş
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/pi/sub_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:

2gradients/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad1^gradients/pi/sub_grad/tuple/control_dependency_1

:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/pi/sub_grad/tuple/control_dependency_13^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@gradients/pi/sub_grad/Reshape_1

<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

gradients/pi/Exp_1_grad/mulMul0gradients/pi/add_1_grad/tuple/control_dependencypi/Exp_1*
T0*
_output_shapes
:
Ţ
'gradients/pi/dense_2/MatMul_grad/MatMulMatMul:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Đ
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(

1gradients/pi/dense_2/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_2/MatMul_grad/MatMul*^gradients/pi/dense_2/MatMul_grad/MatMul_1

9gradients/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_2/MatMul_grad/MatMul2^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_2/MatMul_grad/MatMul_12^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	
Ë
gradients/AddNAddN2gradients/pi/mul_1_grad/tuple/control_dependency_1gradients/pi/Exp_1_grad/mul*
N*
_output_shapes
:*
T0*4
_class*
(&loc:@gradients/pi/mul_1_grad/Reshape_1
˛
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh9gradients/pi/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

2gradients/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_1/Tanh_grad/TanhGrad

:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/Tanh_grad/TanhGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*:
_class0
.,loc:@gradients/pi/dense_1/Tanh_grad/TanhGrad

<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ţ
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ď
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

1gradients/pi/dense_1/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_1/MatMul_grad/MatMul*^gradients/pi/dense_1/MatMul_grad/MatMul_1

9gradients/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/MatMul_grad/MatMul2^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_1/MatMul_grad/MatMul_12^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/pi/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

Ž
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh9gradients/pi/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

0gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp,^gradients/pi/dense/BiasAdd_grad/BiasAddGrad&^gradients/pi/dense/Tanh_grad/TanhGrad

8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/pi/dense/Tanh_grad/TanhGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/pi/dense/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/pi/dense/BiasAdd_grad/BiasAddGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*>
_class4
20loc:@gradients/pi/dense/BiasAdd_grad/BiasAddGrad
×
%gradients/pi/dense/MatMul_grad/MatMulMatMul8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙'*
transpose_a( *
transpose_b(
Č
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder8gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	'*
transpose_a(*
transpose_b( 

/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp&^gradients/pi/dense/MatMul_grad/MatMul(^gradients/pi/dense/MatMul_grad/MatMul_1

7gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity%gradients/pi/dense/MatMul_grad/MatMul0^gradients/pi/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙'*
T0*8
_class.
,*loc:@gradients/pi/dense/MatMul_grad/MatMul

9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity'gradients/pi/dense/MatMul_grad/MatMul_10^gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	'
`
Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

ReshapeReshape9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape/shape*
T0*
Tshape0*
_output_shapes

:
b
Reshape_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_1Reshape:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_1/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_2/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_2Reshape;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_2/shape*
T0*
Tshape0*
_output_shapes

:
b
Reshape_3/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_3Reshape<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_3/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_4/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_4Reshape;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_5/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

	Reshape_5Reshape<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_6/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
h
	Reshape_6Reshapegradients/AddNReshape_6/shape*
T0*
Tshape0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ś
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5	Reshape_6concat/axis*
N*
_output_shapes

:ş	*

Tidx0*
T0
h
PyFuncPyFuncconcat*
_output_shapes

:ş	*
Tin
2*
Tout
2*
token
pyfunc_0
l
Const_5Const*1
value(B&" N                    *
dtype0*
_output_shapes
:
Q
split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 

splitSplitVPyFuncConst_5split/split_dim*
T0*E
_output_shapes3
1:::::::*
	num_split*

Tlen0
`
Reshape_7/shapeConst*
valueB"'      *
dtype0*
_output_shapes
:
d
	Reshape_7ReshapesplitReshape_7/shape*
_output_shapes
:	'*
T0*
Tshape0
Z
Reshape_8/shapeConst*
dtype0*
_output_shapes
:*
valueB:
b
	Reshape_8Reshapesplit:1Reshape_8/shape*
T0*
Tshape0*
_output_shapes	
:
`
Reshape_9/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
g
	Reshape_9Reshapesplit:2Reshape_9/shape* 
_output_shapes
:
*
T0*
Tshape0
[
Reshape_10/shapeConst*
valueB:*
dtype0*
_output_shapes
:
d

Reshape_10Reshapesplit:3Reshape_10/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_11/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
h

Reshape_11Reshapesplit:4Reshape_11/shape*
_output_shapes
:	*
T0*
Tshape0
Z
Reshape_12/shapeConst*
valueB:*
dtype0*
_output_shapes
:
c

Reshape_12Reshapesplit:5Reshape_12/shape*
T0*
Tshape0*
_output_shapes
:
Z
Reshape_13/shapeConst*
valueB:*
dtype0*
_output_shapes
:
c

Reshape_13Reshapesplit:6Reshape_13/shape*
T0*
Tshape0*
_output_shapes
:

beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?* 
_class
loc:@pi/dense/bias

beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape: 
°
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
l
beta1_power/readIdentitybeta1_power*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 

beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wž?* 
_class
loc:@pi/dense/bias

beta2_power
VariableV2*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
°
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
l
beta2_power/readIdentitybeta2_power*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
Ť
6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@pi/dense/kernel*
valueB"'      

,pi/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ô
&pi/dense/kernel/Adam/Initializer/zerosFill6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,pi/dense/kernel/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@pi/dense/kernel*

index_type0*
_output_shapes
:	'
Ž
pi/dense/kernel/Adam
VariableV2*
shape:	'*
dtype0*
_output_shapes
:	'*
shared_name *"
_class
loc:@pi/dense/kernel*
	container 
Ú
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'

pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	'
­
8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@pi/dense/kernel*
valueB"'      *
dtype0*
_output_shapes
:

.pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ú
(pi/dense/kernel/Adam_1/Initializer/zerosFill8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.pi/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@pi/dense/kernel*

index_type0*
_output_shapes
:	'
°
pi/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	'*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	'
ŕ
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel

pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	'

$pi/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
˘
pi/dense/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name * 
_class
loc:@pi/dense/bias
Î
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias

pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:

&pi/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
¤
pi/dense/bias/Adam_1
VariableV2* 
_class
loc:@pi/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ô
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:

pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
_output_shapes	
:*
T0* 
_class
loc:@pi/dense/bias
Ż
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:

.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ý
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@pi/dense_1/kernel*

index_type0* 
_output_shapes
:

´
pi/dense_1/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *$
_class
loc:@pi/dense_1/kernel
ă
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:


pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel
ą
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:

0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*$
_class
loc:@pi/dense_1/kernel*

index_type0
ś
pi/dense_1/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@pi/dense_1/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

é
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:


pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*
T0*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:


&pi/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
pi/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@pi/dense_1/bias*
	container *
shape:
Ö
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias

pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias

(pi/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*"
_class
loc:@pi/dense_1/bias*
valueB*    
¨
pi/dense_1/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@pi/dense_1/bias
Ü
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:

pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
_output_shapes	
:*
T0*"
_class
loc:@pi/dense_1/bias
Ż
8pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

.pi/dense_2/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ü
(pi/dense_2/kernel/Adam/Initializer/zerosFill8pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_2/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@pi/dense_2/kernel*

index_type0*
_output_shapes
:	
˛
pi/dense_2/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container *
shape:	
â
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	

pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	
ą
:pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*$
_class
loc:@pi/dense_2/kernel*
valueB"      

0pi/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*pi/dense_2/kernel/Adam_1/Initializer/zerosFill:pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_2/kernel/Adam_1/Initializer/zeros/Const*
T0*$
_class
loc:@pi/dense_2/kernel*

index_type0*
_output_shapes
:	
´
pi/dense_2/kernel/Adam_1
VariableV2*$
_class
loc:@pi/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
č
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	

&pi/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
¤
pi/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container *
shape:
Ő
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:

pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

(pi/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
Ś
pi/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container *
shape:
Ű
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias

pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:

!pi/log_std/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/log_std*
valueB*    

pi/log_std/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pi/log_std*
	container 
Á
pi/log_std/Adam/AssignAssignpi/log_std/Adam!pi/log_std/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
u
pi/log_std/Adam/readIdentitypi/log_std/Adam*
T0*
_class
loc:@pi/log_std*
_output_shapes
:

#pi/log_std/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@pi/log_std*
valueB*    

pi/log_std/Adam_1
VariableV2*
shared_name *
_class
loc:@pi/log_std*
	container *
shape:*
dtype0*
_output_shapes
:
Ç
pi/log_std/Adam_1/AssignAssignpi/log_std/Adam_1#pi/log_std/Adam_1/Initializer/zeros*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
y
pi/log_std/Adam_1/readIdentitypi/log_std/Adam_1*
T0*
_class
loc:@pi/log_std*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *ŹĹ§7
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ď
%Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_7*
use_locking( *
T0*"
_class
loc:@pi/dense/kernel*
use_nesterov( *
_output_shapes
:	'
Á
#Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_8*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0* 
_class
loc:@pi/dense/bias
Ú
'Adam/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_9*
T0*$
_class
loc:@pi/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
Ě
%Adam/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_10*
T0*"
_class
loc:@pi/dense_1/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ú
'Adam/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_11*
T0*$
_class
loc:@pi/dense_2/kernel*
use_nesterov( *
_output_shapes
:	*
use_locking( 
Ë
%Adam/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_12*
use_locking( *
T0*"
_class
loc:@pi/dense_2/bias*
use_nesterov( *
_output_shapes
:
˛
 Adam/update_pi/log_std/ApplyAdam	ApplyAdam
pi/log_stdpi/log_std/Adampi/log_std/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_13*
use_locking( *
T0*
_class
loc:@pi/log_std*
use_nesterov( *
_output_shapes
:

Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
ż
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam
j
Reshape_14/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
r

Reshape_14Reshapepi/dense/kernel/readReshape_14/shape*
T0*
Tshape0*
_output_shapes

:
j
Reshape_15/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_15Reshapepi/dense/bias/readReshape_15/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_16/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_16Reshapepi/dense_1/kernel/readReshape_16/shape*
T0*
Tshape0*
_output_shapes

:
j
Reshape_17/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_17Reshapepi/dense_1/bias/readReshape_17/shape*
_output_shapes	
:*
T0*
Tshape0
j
Reshape_18/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_18Reshapepi/dense_2/kernel/readReshape_18/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_19/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_19Reshapepi/dense_2/bias/readReshape_19/shape*
T0*
Tshape0*
_output_shapes
:
j
Reshape_20/shapeConst^Adam*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
k

Reshape_20Reshapepi/log_std/readReshape_20/shape*
T0*
Tshape0*
_output_shapes
:
V
concat_1/axisConst^Adam*
value	B : *
dtype0*
_output_shapes
: 
ł
concat_1ConcatV2
Reshape_14
Reshape_15
Reshape_16
Reshape_17
Reshape_18
Reshape_19
Reshape_20concat_1/axis*
T0*
N*
_output_shapes

:ş	*

Tidx0
h
PyFunc_1PyFuncconcat_1*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_1
s
Const_6Const^Adam*
dtype0*
_output_shapes
:*1
value(B&" N                    
Z
split_1/split_dimConst^Adam*
value	B : *
dtype0*
_output_shapes
: 

split_1SplitVPyFunc_1Const_6split_1/split_dim*0
_output_shapes
:::::::*
	num_split*

Tlen0*
T0
h
Reshape_21/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB"'      
h

Reshape_21Reshapesplit_1Reshape_21/shape*
T0*
Tshape0*
_output_shapes
:	'
b
Reshape_22/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_22Reshape	split_1:1Reshape_22/shape*
T0*
Tshape0*
_output_shapes	
:
h
Reshape_23/shapeConst^Adam*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_23Reshape	split_1:2Reshape_23/shape*
T0*
Tshape0* 
_output_shapes
:

b
Reshape_24/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_24Reshape	split_1:3Reshape_24/shape*
T0*
Tshape0*
_output_shapes	
:
h
Reshape_25/shapeConst^Adam*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_25Reshape	split_1:4Reshape_25/shape*
T0*
Tshape0*
_output_shapes
:	
a
Reshape_26/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_26Reshape	split_1:5Reshape_26/shape*
T0*
Tshape0*
_output_shapes
:
a
Reshape_27/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_27Reshape	split_1:6Reshape_27/shape*
T0*
Tshape0*
_output_shapes
:
¤
AssignAssignpi/dense/kernel
Reshape_21*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(

Assign_1Assignpi/dense/bias
Reshape_22*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ť
Assign_2Assignpi/dense_1/kernel
Reshape_23*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˘
Assign_3Assignpi/dense_1/bias
Reshape_24*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
Ş
Assign_4Assignpi/dense_2/kernel
Reshape_25*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ą
Assign_5Assignpi/dense_2/bias
Reshape_26*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias

Assign_6Assign
pi/log_std
Reshape_27*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
d

group_depsNoOp^Adam^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6
(
group_deps_1NoOp^Adam^group_deps
T
gradients_1/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
gradients_1/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients_1/Mean_1_grad/ShapeShapepow*
T0*
out_type0*
_output_shapes
:
¤
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
˘
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ś
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
c
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_1/pow_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients_1/pow_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
ş
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_1/pow_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
T0*
_output_shapes
: 
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients_1/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients_1/pow_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
˛
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0

gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
Ţ
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1*
_output_shapes
: 
i
gradients_1/sub_1_grad/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
g
gradients_1/sub_1_grad/Shape_1Shape	v/Squeeze*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ž
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
gradients_1/sub_1_grad/Sum_1Sum-gradients_1/pow_grad/tuple/control_dependency.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
T0*
_output_shapes
:
Ł
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ć
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
ě
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
 gradients_1/v/Squeeze_grad/ShapeShapev/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
Â
"gradients_1/v/Squeeze_grad/ReshapeReshape1gradients_1/sub_1_grad/tuple/control_dependency_1 gradients_1/v/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients_1/v/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

3gradients_1/v/dense_2/BiasAdd_grad/tuple/group_depsNoOp#^gradients_1/v/Squeeze_grad/Reshape/^gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad

;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients_1/v/Squeeze_grad/Reshape4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*5
_class+
)'loc:@gradients_1/v/Squeeze_grad/Reshape

=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ß
(gradients_1/v/dense_2/MatMul_grad/MatMulMatMul;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyv/dense_2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ń
*gradients_1/v/dense_2/MatMul_grad/MatMul_1MatMulv/dense_1/Tanh;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

2gradients_1/v/dense_2/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_2/MatMul_grad/MatMul+^gradients_1/v/dense_2/MatMul_grad/MatMul_1

:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_2/MatMul_grad/MatMul3^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_2/MatMul_grad/MatMul_13^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*=
_class3
1/loc:@gradients_1/v/dense_2/MatMul_grad/MatMul_1
ł
(gradients_1/v/dense_1/Tanh_grad/TanhGradTanhGradv/dense_1/Tanh:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

3gradients_1/v/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_1/Tanh_grad/TanhGrad

;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/Tanh_grad/TanhGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ß
(gradients_1/v/dense_1/MatMul_grad/MatMulMatMul;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyv/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Đ
*gradients_1/v/dense_1/MatMul_grad/MatMul_1MatMulv/dense/Tanh;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

2gradients_1/v/dense_1/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_1/MatMul_grad/MatMul+^gradients_1/v/dense_1/MatMul_grad/MatMul_1

:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/MatMul_grad/MatMul3^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_1/MatMul_grad/MatMul_13^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*=
_class3
1/loc:@gradients_1/v/dense_1/MatMul_grad/MatMul_1
Ż
&gradients_1/v/dense/Tanh_grad/TanhGradTanhGradv/dense/Tanh:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
,gradients_1/v/dense/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/v/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:

1gradients_1/v/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients_1/v/dense/BiasAdd_grad/BiasAddGrad'^gradients_1/v/dense/Tanh_grad/TanhGrad

9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/Tanh_grad/TanhGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*9
_class/
-+loc:@gradients_1/v/dense/Tanh_grad/TanhGrad

;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients_1/v/dense/BiasAdd_grad/BiasAddGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/v/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ř
&gradients_1/v/dense/MatMul_grad/MatMulMatMul9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyv/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙'*
transpose_a( *
transpose_b(
Ę
(gradients_1/v/dense/MatMul_grad/MatMul_1MatMulPlaceholder9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	'*
transpose_a(*
transpose_b( 

0gradients_1/v/dense/MatMul_grad/tuple/group_depsNoOp'^gradients_1/v/dense/MatMul_grad/MatMul)^gradients_1/v/dense/MatMul_grad/MatMul_1

8gradients_1/v/dense/MatMul_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/MatMul_grad/MatMul1^gradients_1/v/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙'*
T0*9
_class/
-+loc:@gradients_1/v/dense/MatMul_grad/MatMul

:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Identity(gradients_1/v/dense/MatMul_grad/MatMul_11^gradients_1/v/dense/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense/MatMul_grad/MatMul_1*
_output_shapes
:	'
c
Reshape_28/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_28Reshape:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Reshape_28/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_29/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_29Reshape;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_29/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_30/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_30Reshape<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_30/shape*
_output_shapes

:*
T0*
Tshape0
c
Reshape_31/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_31Reshape=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_31/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_32/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙


Reshape_32Reshape<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_32/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_33/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:


Reshape_33Reshape=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_33/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
§
concat_2ConcatV2
Reshape_28
Reshape_29
Reshape_30
Reshape_31
Reshape_32
Reshape_33concat_2/axis*
T0*
N*
_output_shapes

:¤	*

Tidx0
l
PyFunc_2PyFuncconcat_2*
_output_shapes

:¤	*
Tin
2*
Tout
2*
token
pyfunc_2
h
Const_7Const*-
value$B"" N                 *
dtype0*
_output_shapes
:
S
split_2/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 

split_2SplitVPyFunc_2Const_7split_2/split_dim*?
_output_shapes-
+::::::*
	num_split*

Tlen0*
T0
a
Reshape_34/shapeConst*
valueB"'      *
dtype0*
_output_shapes
:
h

Reshape_34Reshapesplit_2Reshape_34/shape*
T0*
Tshape0*
_output_shapes
:	'
[
Reshape_35/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_35Reshape	split_2:1Reshape_35/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_36/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_36Reshape	split_2:2Reshape_36/shape*
T0*
Tshape0* 
_output_shapes
:

[
Reshape_37/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_37Reshape	split_2:3Reshape_37/shape*
T0*
Tshape0*
_output_shapes	
:
a
Reshape_38/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_38Reshape	split_2:4Reshape_38/shape*
T0*
Tshape0*
_output_shapes
:	
Z
Reshape_39/shapeConst*
dtype0*
_output_shapes
:*
valueB:
e

Reshape_39Reshape	split_2:5Reshape_39/shape*
T0*
Tshape0*
_output_shapes
:

beta1_power_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@v/dense/bias

beta1_power_1
VariableV2*
shared_name *
_class
loc:@v/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ľ
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
o
beta1_power_1/readIdentitybeta1_power_1*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 

beta2_power_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wž?*
_class
loc:@v/dense/bias

beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@v/dense/bias*
	container *
shape: 
ľ
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
o
beta2_power_1/readIdentitybeta2_power_1*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
Š
5v/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*!
_class
loc:@v/dense/kernel*
valueB"'      

+v/dense/kernel/Adam/Initializer/zeros/ConstConst*!
_class
loc:@v/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
đ
%v/dense/kernel/Adam/Initializer/zerosFill5v/dense/kernel/Adam/Initializer/zeros/shape_as_tensor+v/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	'*
T0*!
_class
loc:@v/dense/kernel*

index_type0
Ź
v/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	'*
shared_name *!
_class
loc:@v/dense/kernel*
	container *
shape:	'
Ö
v/dense/kernel/Adam/AssignAssignv/dense/kernel/Adam%v/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'

v/dense/kernel/Adam/readIdentityv/dense/kernel/Adam*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	'
Ť
7v/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@v/dense/kernel*
valueB"'      *
dtype0*
_output_shapes
:

-v/dense/kernel/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@v/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ö
'v/dense/kernel/Adam_1/Initializer/zerosFill7v/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor-v/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*!
_class
loc:@v/dense/kernel*

index_type0*
_output_shapes
:	'
Ž
v/dense/kernel/Adam_1
VariableV2*
shared_name *!
_class
loc:@v/dense/kernel*
	container *
shape:	'*
dtype0*
_output_shapes
:	'
Ü
v/dense/kernel/Adam_1/AssignAssignv/dense/kernel/Adam_1'v/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel

v/dense/kernel/Adam_1/readIdentityv/dense/kernel/Adam_1*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	'

#v/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@v/dense/bias*
valueB*    
 
v/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@v/dense/bias*
	container *
shape:
Ę
v/dense/bias/Adam/AssignAssignv/dense/bias/Adam#v/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
|
v/dense/bias/Adam/readIdentityv/dense/bias/Adam*
T0*
_class
loc:@v/dense/bias*
_output_shapes	
:

%v/dense/bias/Adam_1/Initializer/zerosConst*
_class
loc:@v/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
˘
v/dense/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@v/dense/bias*
	container 
Đ
v/dense/bias/Adam_1/AssignAssignv/dense/bias/Adam_1%v/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias

v/dense/bias/Adam_1/readIdentityv/dense/bias/Adam_1*
T0*
_class
loc:@v/dense/bias*
_output_shapes	
:
­
7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@v/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:

-v/dense_1/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@v/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ů
'v/dense_1/kernel/Adam/Initializer/zerosFill7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor-v/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@v/dense_1/kernel*

index_type0* 
_output_shapes
:

˛
v/dense_1/kernel/Adam
VariableV2*
shared_name *#
_class
loc:@v/dense_1/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ß
v/dense_1/kernel/Adam/AssignAssignv/dense_1/kernel/Adam'v/dense_1/kernel/Adam/Initializer/zeros*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

v/dense_1/kernel/Adam/readIdentityv/dense_1/kernel/Adam*
T0*#
_class
loc:@v/dense_1/kernel* 
_output_shapes
:

Ż
9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*#
_class
loc:@v/dense_1/kernel*
valueB"      

/v/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@v/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
˙
)v/dense_1/kernel/Adam_1/Initializer/zerosFill9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor/v/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@v/dense_1/kernel*

index_type0* 
_output_shapes
:

´
v/dense_1/kernel/Adam_1
VariableV2*
shared_name *#
_class
loc:@v/dense_1/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ĺ
v/dense_1/kernel/Adam_1/AssignAssignv/dense_1/kernel/Adam_1)v/dense_1/kernel/Adam_1/Initializer/zeros*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

v/dense_1/kernel/Adam_1/readIdentityv/dense_1/kernel/Adam_1* 
_output_shapes
:
*
T0*#
_class
loc:@v/dense_1/kernel

%v/dense_1/bias/Adam/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
¤
v/dense_1/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *!
_class
loc:@v/dense_1/bias
Ň
v/dense_1/bias/Adam/AssignAssignv/dense_1/bias/Adam%v/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias

v/dense_1/bias/Adam/readIdentityv/dense_1/bias/Adam*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes	
:

'v/dense_1/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
v/dense_1/bias/Adam_1
VariableV2*
shared_name *!
_class
loc:@v/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ř
v/dense_1/bias/Adam_1/AssignAssignv/dense_1/bias/Adam_1'v/dense_1/bias/Adam_1/Initializer/zeros*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

v/dense_1/bias/Adam_1/readIdentityv/dense_1/bias/Adam_1*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes	
:
Ł
'v/dense_2/kernel/Adam/Initializer/zerosConst*#
_class
loc:@v/dense_2/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
°
v/dense_2/kernel/Adam
VariableV2*
shared_name *#
_class
loc:@v/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ţ
v/dense_2/kernel/Adam/AssignAssignv/dense_2/kernel/Adam'v/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel

v/dense_2/kernel/Adam/readIdentityv/dense_2/kernel/Adam*
_output_shapes
:	*
T0*#
_class
loc:@v/dense_2/kernel
Ľ
)v/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	*#
_class
loc:@v/dense_2/kernel*
valueB	*    
˛
v/dense_2/kernel/Adam_1
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *#
_class
loc:@v/dense_2/kernel
ä
v/dense_2/kernel/Adam_1/AssignAssignv/dense_2/kernel/Adam_1)v/dense_2/kernel/Adam_1/Initializer/zeros*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

v/dense_2/kernel/Adam_1/readIdentityv/dense_2/kernel/Adam_1*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes
:	

%v/dense_2/bias/Adam/Initializer/zerosConst*!
_class
loc:@v/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
˘
v/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@v/dense_2/bias*
	container *
shape:
Ń
v/dense_2/bias/Adam/AssignAssignv/dense_2/bias/Adam%v/dense_2/bias/Adam/Initializer/zeros*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(

v/dense_2/bias/Adam/readIdentityv/dense_2/bias/Adam*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:

'v/dense_2/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@v/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
¤
v/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@v/dense_2/bias*
	container *
shape:
×
v/dense_2/bias/Adam_1/AssignAssignv/dense_2/bias/Adam_1'v/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:

v/dense_2/bias/Adam_1/readIdentityv/dense_2/bias/Adam_1*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
Y
Adam_1/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *
×#<
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ů
&Adam_1/update_v/dense/kernel/ApplyAdam	ApplyAdamv/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_34*
use_locking( *
T0*!
_class
loc:@v/dense/kernel*
use_nesterov( *
_output_shapes
:	'
Ë
$Adam_1/update_v/dense/bias/ApplyAdam	ApplyAdamv/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_35*
use_locking( *
T0*
_class
loc:@v/dense/bias*
use_nesterov( *
_output_shapes	
:
ä
(Adam_1/update_v/dense_1/kernel/ApplyAdam	ApplyAdamv/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_36*
use_locking( *
T0*#
_class
loc:@v/dense_1/kernel*
use_nesterov( * 
_output_shapes
:

Ő
&Adam_1/update_v/dense_1/bias/ApplyAdam	ApplyAdamv/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_37*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*!
_class
loc:@v/dense_1/bias
ă
(Adam_1/update_v/dense_2/kernel/ApplyAdam	ApplyAdamv/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_38*
use_locking( *
T0*#
_class
loc:@v/dense_2/kernel*
use_nesterov( *
_output_shapes
:	
Ô
&Adam_1/update_v/dense_2/bias/ApplyAdam	ApplyAdamv/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_39*
T0*!
_class
loc:@v/dense_2/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
í

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 

Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@v/dense/bias
ď
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
Ą
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
¨
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam
l
Reshape_40/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_40Reshapev/dense/kernel/readReshape_40/shape*
T0*
Tshape0*
_output_shapes

:
l
Reshape_41/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n

Reshape_41Reshapev/dense/bias/readReshape_41/shape*
_output_shapes	
:*
T0*
Tshape0
l
Reshape_42/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_42Reshapev/dense_1/kernel/readReshape_42/shape*
_output_shapes

:*
T0*
Tshape0
l
Reshape_43/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_43Reshapev/dense_1/bias/readReshape_43/shape*
T0*
Tshape0*
_output_shapes	
:
l
Reshape_44/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r

Reshape_44Reshapev/dense_2/kernel/readReshape_44/shape*
T0*
Tshape0*
_output_shapes	
:
l
Reshape_45/shapeConst^Adam_1*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_45Reshapev/dense_2/bias/readReshape_45/shape*
T0*
Tshape0*
_output_shapes
:
X
concat_3/axisConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 
§
concat_3ConcatV2
Reshape_40
Reshape_41
Reshape_42
Reshape_43
Reshape_44
Reshape_45concat_3/axis*
N*
_output_shapes

:¤	*

Tidx0*
T0
h
PyFunc_3PyFuncconcat_3*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_3
q
Const_8Const^Adam_1*-
value$B"" N                 *
dtype0*
_output_shapes
:
\
split_3/split_dimConst^Adam_1*
value	B : *
dtype0*
_output_shapes
: 

split_3SplitVPyFunc_3Const_8split_3/split_dim*
T0*,
_output_shapes
::::::*
	num_split*

Tlen0
j
Reshape_46/shapeConst^Adam_1*
valueB"'      *
dtype0*
_output_shapes
:
h

Reshape_46Reshapesplit_3Reshape_46/shape*
_output_shapes
:	'*
T0*
Tshape0
d
Reshape_47/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_47Reshape	split_3:1Reshape_47/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_48/shapeConst^Adam_1*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_48Reshape	split_3:2Reshape_48/shape*
T0*
Tshape0* 
_output_shapes
:

d
Reshape_49/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_49Reshape	split_3:3Reshape_49/shape*
T0*
Tshape0*
_output_shapes	
:
j
Reshape_50/shapeConst^Adam_1*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_50Reshape	split_3:4Reshape_50/shape*
T0*
Tshape0*
_output_shapes
:	
c
Reshape_51/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
e

Reshape_51Reshape	split_3:5Reshape_51/shape*
_output_shapes
:*
T0*
Tshape0
¤
Assign_7Assignv/dense/kernel
Reshape_46*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(

Assign_8Assignv/dense/bias
Reshape_47*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
Š
Assign_9Assignv/dense_1/kernel
Reshape_48*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ą
	Assign_10Assignv/dense_1/bias
Reshape_49*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
Š
	Assign_11Assignv/dense_2/kernel
Reshape_50*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
 
	Assign_12Assignv/dense_2/bias
Reshape_51*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
b
group_deps_2NoOp^Adam_1
^Assign_10
^Assign_11
^Assign_12	^Assign_7	^Assign_8	^Assign_9
,
group_deps_3NoOp^Adam_1^group_deps_2
ş	
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^pi/log_std/Adam/Assign^pi/log_std/Adam_1/Assign^pi/log_std/Assign^v/dense/bias/Adam/Assign^v/dense/bias/Adam_1/Assign^v/dense/bias/Assign^v/dense/kernel/Adam/Assign^v/dense/kernel/Adam_1/Assign^v/dense/kernel/Assign^v/dense_1/bias/Adam/Assign^v/dense_1/bias/Adam_1/Assign^v/dense_1/bias/Assign^v/dense_1/kernel/Adam/Assign^v/dense_1/kernel/Adam_1/Assign^v/dense_1/kernel/Assign^v/dense_2/bias/Adam/Assign^v/dense_2/bias/Adam_1/Assign^v/dense_2/bias/Assign^v/dense_2/kernel/Adam/Assign^v/dense_2/kernel/Adam_1/Assign^v/dense_2/kernel/Assign
c
Reshape_52/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r

Reshape_52Reshapepi/dense/kernel/readReshape_52/shape*
_output_shapes

:*
T0*
Tshape0
c
Reshape_53/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_53Reshapepi/dense/bias/readReshape_53/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_54/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_54Reshapepi/dense_1/kernel/readReshape_54/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_55/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q

Reshape_55Reshapepi/dense_1/bias/readReshape_55/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_56/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_56Reshapepi/dense_2/kernel/readReshape_56/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_57/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_57Reshapepi/dense_2/bias/readReshape_57/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_58/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
k

Reshape_58Reshapepi/log_std/readReshape_58/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_59/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
q

Reshape_59Reshapev/dense/kernel/readReshape_59/shape*
_output_shapes

:*
T0*
Tshape0
c
Reshape_60/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n

Reshape_60Reshapev/dense/bias/readReshape_60/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_61/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_61Reshapev/dense_1/kernel/readReshape_61/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_62/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
p

Reshape_62Reshapev/dense_1/bias/readReshape_62/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_63/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
r

Reshape_63Reshapev/dense_2/kernel/readReshape_63/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_64/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o

Reshape_64Reshapev/dense_2/bias/readReshape_64/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_65/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
l

Reshape_65Reshapebeta1_power/readReshape_65/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_66/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
l

Reshape_66Reshapebeta2_power/readReshape_66/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_67/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w

Reshape_67Reshapepi/dense/kernel/Adam/readReshape_67/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_68/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y

Reshape_68Reshapepi/dense/kernel/Adam_1/readReshape_68/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_69/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
t

Reshape_69Reshapepi/dense/bias/Adam/readReshape_69/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_70/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
v

Reshape_70Reshapepi/dense/bias/Adam_1/readReshape_70/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_71/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y

Reshape_71Reshapepi/dense_1/kernel/Adam/readReshape_71/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_72/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{

Reshape_72Reshapepi/dense_1/kernel/Adam_1/readReshape_72/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_73/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
v

Reshape_73Reshapepi/dense_1/bias/Adam/readReshape_73/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_74/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x

Reshape_74Reshapepi/dense_1/bias/Adam_1/readReshape_74/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_75/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x

Reshape_75Reshapepi/dense_2/kernel/Adam/readReshape_75/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_76/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z

Reshape_76Reshapepi/dense_2/kernel/Adam_1/readReshape_76/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_77/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
u

Reshape_77Reshapepi/dense_2/bias/Adam/readReshape_77/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_78/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w

Reshape_78Reshapepi/dense_2/bias/Adam_1/readReshape_78/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_79/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
p

Reshape_79Reshapepi/log_std/Adam/readReshape_79/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_80/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
r

Reshape_80Reshapepi/log_std/Adam_1/readReshape_80/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_81/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n

Reshape_81Reshapebeta1_power_1/readReshape_81/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_82/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
n

Reshape_82Reshapebeta2_power_1/readReshape_82/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_83/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
v

Reshape_83Reshapev/dense/kernel/Adam/readReshape_83/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_84/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x

Reshape_84Reshapev/dense/kernel/Adam_1/readReshape_84/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_85/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s

Reshape_85Reshapev/dense/bias/Adam/readReshape_85/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_86/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
u

Reshape_86Reshapev/dense/bias/Adam_1/readReshape_86/shape*
_output_shapes	
:*
T0*
Tshape0
c
Reshape_87/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
x

Reshape_87Reshapev/dense_1/kernel/Adam/readReshape_87/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_88/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
z

Reshape_88Reshapev/dense_1/kernel/Adam_1/readReshape_88/shape*
T0*
Tshape0*
_output_shapes

:
c
Reshape_89/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
u

Reshape_89Reshapev/dense_1/bias/Adam/readReshape_89/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_90/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w

Reshape_90Reshapev/dense_1/bias/Adam_1/readReshape_90/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_91/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w

Reshape_91Reshapev/dense_2/kernel/Adam/readReshape_91/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_92/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
y

Reshape_92Reshapev/dense_2/kernel/Adam_1/readReshape_92/shape*
T0*
Tshape0*
_output_shapes	
:
c
Reshape_93/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
t

Reshape_93Reshapev/dense_2/bias/Adam/readReshape_93/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_94/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
v

Reshape_94Reshapev/dense_2/bias/Adam_1/readReshape_94/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_4/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ă
concat_4ConcatV2
Reshape_52
Reshape_53
Reshape_54
Reshape_55
Reshape_56
Reshape_57
Reshape_58
Reshape_59
Reshape_60
Reshape_61
Reshape_62
Reshape_63
Reshape_64
Reshape_65
Reshape_66
Reshape_67
Reshape_68
Reshape_69
Reshape_70
Reshape_71
Reshape_72
Reshape_73
Reshape_74
Reshape_75
Reshape_76
Reshape_77
Reshape_78
Reshape_79
Reshape_80
Reshape_81
Reshape_82
Reshape_83
Reshape_84
Reshape_85
Reshape_86
Reshape_87
Reshape_88
Reshape_89
Reshape_90
Reshape_91
Reshape_92
Reshape_93
Reshape_94concat_4/axis*
T0*
N+*
_output_shapes

:Ď8*

Tidx0
h
PyFunc_4PyFuncconcat_4*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_4

Const_9Const*Ä
valueşBˇ+"Ź N                     N                        N   N                                             N   N                                *
dtype0*
_output_shapes
:+
S
split_4/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
˘
split_4SplitVPyFunc_4Const_9split_4/split_dim*
T0*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*
	num_split+*

Tlen0
a
Reshape_95/shapeConst*
valueB"'      *
dtype0*
_output_shapes
:
h

Reshape_95Reshapesplit_4Reshape_95/shape*
T0*
Tshape0*
_output_shapes
:	'
[
Reshape_96/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_96Reshape	split_4:1Reshape_96/shape*
_output_shapes	
:*
T0*
Tshape0
a
Reshape_97/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_97Reshape	split_4:2Reshape_97/shape*
T0*
Tshape0* 
_output_shapes
:

[
Reshape_98/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_98Reshape	split_4:3Reshape_98/shape*
T0*
Tshape0*
_output_shapes	
:
a
Reshape_99/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_99Reshape	split_4:4Reshape_99/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_100/shapeConst*
dtype0*
_output_shapes
:*
valueB:
g
Reshape_100Reshape	split_4:5Reshape_100/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_101/shapeConst*
dtype0*
_output_shapes
:*
valueB:
g
Reshape_101Reshape	split_4:6Reshape_101/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_102/shapeConst*
dtype0*
_output_shapes
:*
valueB"'      
l
Reshape_102Reshape	split_4:7Reshape_102/shape*
T0*
Tshape0*
_output_shapes
:	'
\
Reshape_103/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_103Reshape	split_4:8Reshape_103/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_104/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_104Reshape	split_4:9Reshape_104/shape*
T0*
Tshape0* 
_output_shapes
:

\
Reshape_105/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_105Reshape
split_4:10Reshape_105/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_106/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_106Reshape
split_4:11Reshape_106/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_107/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_107Reshape
split_4:12Reshape_107/shape*
T0*
Tshape0*
_output_shapes
:
T
Reshape_108/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_108Reshape
split_4:13Reshape_108/shape*
T0*
Tshape0*
_output_shapes
: 
T
Reshape_109/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_109Reshape
split_4:14Reshape_109/shape*
T0*
Tshape0*
_output_shapes
: 
b
Reshape_110/shapeConst*
dtype0*
_output_shapes
:*
valueB"'      
m
Reshape_110Reshape
split_4:15Reshape_110/shape*
T0*
Tshape0*
_output_shapes
:	'
b
Reshape_111/shapeConst*
dtype0*
_output_shapes
:*
valueB"'      
m
Reshape_111Reshape
split_4:16Reshape_111/shape*
_output_shapes
:	'*
T0*
Tshape0
\
Reshape_112/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_112Reshape
split_4:17Reshape_112/shape*
_output_shapes	
:*
T0*
Tshape0
\
Reshape_113/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_113Reshape
split_4:18Reshape_113/shape*
_output_shapes	
:*
T0*
Tshape0
b
Reshape_114/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_114Reshape
split_4:19Reshape_114/shape*
T0*
Tshape0* 
_output_shapes
:

b
Reshape_115/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_115Reshape
split_4:20Reshape_115/shape*
T0*
Tshape0* 
_output_shapes
:

\
Reshape_116/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_116Reshape
split_4:21Reshape_116/shape*
T0*
Tshape0*
_output_shapes	
:
\
Reshape_117/shapeConst*
dtype0*
_output_shapes
:*
valueB:
i
Reshape_117Reshape
split_4:22Reshape_117/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_118/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_118Reshape
split_4:23Reshape_118/shape*
T0*
Tshape0*
_output_shapes
:	
b
Reshape_119/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_119Reshape
split_4:24Reshape_119/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_120/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_120Reshape
split_4:25Reshape_120/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_121/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_121Reshape
split_4:26Reshape_121/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_122/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_122Reshape
split_4:27Reshape_122/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_123/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_123Reshape
split_4:28Reshape_123/shape*
T0*
Tshape0*
_output_shapes
:
T
Reshape_124/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_124Reshape
split_4:29Reshape_124/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_125/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_125Reshape
split_4:30Reshape_125/shape*
T0*
Tshape0*
_output_shapes
: 
b
Reshape_126/shapeConst*
valueB"'      *
dtype0*
_output_shapes
:
m
Reshape_126Reshape
split_4:31Reshape_126/shape*
T0*
Tshape0*
_output_shapes
:	'
b
Reshape_127/shapeConst*
valueB"'      *
dtype0*
_output_shapes
:
m
Reshape_127Reshape
split_4:32Reshape_127/shape*
_output_shapes
:	'*
T0*
Tshape0
\
Reshape_128/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_128Reshape
split_4:33Reshape_128/shape*
T0*
Tshape0*
_output_shapes	
:
\
Reshape_129/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_129Reshape
split_4:34Reshape_129/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_130/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_130Reshape
split_4:35Reshape_130/shape*
T0*
Tshape0* 
_output_shapes
:

b
Reshape_131/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_131Reshape
split_4:36Reshape_131/shape*
T0*
Tshape0* 
_output_shapes
:

\
Reshape_132/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_132Reshape
split_4:37Reshape_132/shape*
T0*
Tshape0*
_output_shapes	
:
\
Reshape_133/shapeConst*
valueB:*
dtype0*
_output_shapes
:
i
Reshape_133Reshape
split_4:38Reshape_133/shape*
T0*
Tshape0*
_output_shapes	
:
b
Reshape_134/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_134Reshape
split_4:39Reshape_134/shape*
_output_shapes
:	*
T0*
Tshape0
b
Reshape_135/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
m
Reshape_135Reshape
split_4:40Reshape_135/shape*
T0*
Tshape0*
_output_shapes
:	
[
Reshape_136/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_136Reshape
split_4:41Reshape_136/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_137/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_137Reshape
split_4:42Reshape_137/shape*
_output_shapes
:*
T0*
Tshape0
§
	Assign_13Assignpi/dense/kernel
Reshape_95*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(

	Assign_14Assignpi/dense/bias
Reshape_96*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
Ź
	Assign_15Assignpi/dense_1/kernel
Reshape_97*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ł
	Assign_16Assignpi/dense_1/bias
Reshape_98*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
Ť
	Assign_17Assignpi/dense_2/kernel
Reshape_99*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ł
	Assign_18Assignpi/dense_2/biasReshape_100*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:

	Assign_19Assign
pi/log_stdReshape_101*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ś
	Assign_20Assignv/dense/kernelReshape_102*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(

	Assign_21Assignv/dense/biasReshape_103*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
Ť
	Assign_22Assignv/dense_1/kernelReshape_104*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˘
	Assign_23Assignv/dense_1/biasReshape_105*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
Ş
	Assign_24Assignv/dense_2/kernelReshape_106*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Ą
	Assign_25Assignv/dense_2/biasReshape_107*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(

	Assign_26Assignbeta1_powerReshape_108*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 

	Assign_27Assignbeta2_powerReshape_109*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
­
	Assign_28Assignpi/dense/kernel/AdamReshape_110*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
Ż
	Assign_29Assignpi/dense/kernel/Adam_1Reshape_111*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
Ľ
	Assign_30Assignpi/dense/bias/AdamReshape_112*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
§
	Assign_31Assignpi/dense/bias/Adam_1Reshape_113*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
	Assign_32Assignpi/dense_1/kernel/AdamReshape_114*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
´
	Assign_33Assignpi/dense_1/kernel/Adam_1Reshape_115*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Š
	Assign_34Assignpi/dense_1/bias/AdamReshape_116*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
Ť
	Assign_35Assignpi/dense_1/bias/Adam_1Reshape_117*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ą
	Assign_36Assignpi/dense_2/kernel/AdamReshape_118*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
ł
	Assign_37Assignpi/dense_2/kernel/Adam_1Reshape_119*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
¨
	Assign_38Assignpi/dense_2/bias/AdamReshape_120*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
Ş
	Assign_39Assignpi/dense_2/bias/Adam_1Reshape_121*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:

	Assign_40Assignpi/log_std/AdamReshape_122*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
 
	Assign_41Assignpi/log_std/Adam_1Reshape_123*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std

	Assign_42Assignbeta1_power_1Reshape_124*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 

	Assign_43Assignbeta2_power_1Reshape_125*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ť
	Assign_44Assignv/dense/kernel/AdamReshape_126*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
­
	Assign_45Assignv/dense/kernel/Adam_1Reshape_127*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
Ł
	Assign_46Assignv/dense/bias/AdamReshape_128*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
Ľ
	Assign_47Assignv/dense/bias/Adam_1Reshape_129*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
°
	Assign_48Assignv/dense_1/kernel/AdamReshape_130*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
˛
	Assign_49Assignv/dense_1/kernel/Adam_1Reshape_131*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
§
	Assign_50Assignv/dense_1/bias/AdamReshape_132*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
Š
	Assign_51Assignv/dense_1/bias/Adam_1Reshape_133*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
Ż
	Assign_52Assignv/dense_2/kernel/AdamReshape_134*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ą
	Assign_53Assignv/dense_2/kernel/Adam_1Reshape_135*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Ś
	Assign_54Assignv/dense_2/bias/AdamReshape_136*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
¨
	Assign_55Assignv/dense_2/bias/Adam_1Reshape_137*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:

group_deps_4NoOp
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_9144ed3cd68e4a83b0e5d5eaf1efdd8d/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
¸
save/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
š
save/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ô
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
ť
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:+*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
ź
save/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ĺ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+

save/AssignAssignbeta1_powersave/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ł
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
˘
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ł
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Š
save/Assign_4Assignpi/dense/biassave/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
Ž
save/Assign_5Assignpi/dense/bias/Adamsave/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
°
save/Assign_6Assignpi/dense/bias/Adam_1save/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ą
save/Assign_7Assignpi/dense/kernelsave/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ś
save/Assign_8Assignpi/dense/kernel/Adamsave/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
¸
save/Assign_9Assignpi/dense/kernel/Adam_1save/RestoreV2:9*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
Ż
save/Assign_10Assignpi/dense_1/biassave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
´
save/Assign_11Assignpi/dense_1/bias/Adamsave/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ś
save/Assign_12Assignpi/dense_1/bias/Adam_1save/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save/Assign_13Assignpi/dense_1/kernelsave/RestoreV2:13*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˝
save/Assign_14Assignpi/dense_1/kernel/Adamsave/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save/Assign_15Assignpi/dense_1/kernel/Adam_1save/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ž
save/Assign_16Assignpi/dense_2/biassave/RestoreV2:16*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ł
save/Assign_17Assignpi/dense_2/bias/Adamsave/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ľ
save/Assign_18Assignpi/dense_2/bias/Adam_1save/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save/Assign_19Assignpi/dense_2/kernelsave/RestoreV2:19*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
ź
save/Assign_20Assignpi/dense_2/kernel/Adamsave/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ž
save/Assign_21Assignpi/dense_2/kernel/Adam_1save/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
¤
save/Assign_22Assign
pi/log_stdsave/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
Š
save/Assign_23Assignpi/log_std/Adamsave/RestoreV2:23*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ť
save/Assign_24Assignpi/log_std/Adam_1save/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Š
save/Assign_25Assignv/dense/biassave/RestoreV2:25*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
Ž
save/Assign_26Assignv/dense/bias/Adamsave/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
°
save/Assign_27Assignv/dense/bias/Adam_1save/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ą
save/Assign_28Assignv/dense/kernelsave/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ś
save/Assign_29Assignv/dense/kernel/Adamsave/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
¸
save/Assign_30Assignv/dense/kernel/Adam_1save/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
­
save/Assign_31Assignv/dense_1/biassave/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
˛
save/Assign_32Assignv/dense_1/bias/Adamsave/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
´
save/Assign_33Assignv/dense_1/bias/Adam_1save/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ś
save/Assign_34Assignv/dense_1/kernelsave/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ť
save/Assign_35Assignv/dense_1/kernel/Adamsave/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˝
save/Assign_36Assignv/dense_1/kernel/Adam_1save/RestoreV2:36*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ź
save/Assign_37Assignv/dense_2/biassave/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ą
save/Assign_38Assignv/dense_2/bias/Adamsave/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ł
save/Assign_39Assignv/dense_2/bias/Adam_1save/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ľ
save/Assign_40Assignv/dense_2/kernelsave/RestoreV2:40*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
ş
save/Assign_41Assignv/dense_2/kernel/Adamsave/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ź
save/Assign_42Assignv/dense_2/kernel/Adam_1save/RestoreV2:42*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
é
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_8d359a7e5db74f8e86aa68ec0700e0f8/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
ş
save_1/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ť
save_1/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ü
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
Ł
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
˝
save_1/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_1/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
í
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
˘
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
§
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ś
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
§
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
­
save_1/Assign_4Assignpi/dense/biassave_1/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
˛
save_1/Assign_5Assignpi/dense/bias/Adamsave_1/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_1/Assign_6Assignpi/dense/bias/Adam_1save_1/RestoreV2:6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ľ
save_1/Assign_7Assignpi/dense/kernelsave_1/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ş
save_1/Assign_8Assignpi/dense/kernel/Adamsave_1/RestoreV2:8*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ź
save_1/Assign_9Assignpi/dense/kernel/Adam_1save_1/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_1/Assign_10Assignpi/dense_1/biassave_1/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_1/Assign_11Assignpi/dense_1/bias/Adamsave_1/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_1/Assign_12Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_1/Assign_13Assignpi/dense_1/kernelsave_1/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_1/Assign_14Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_1/Assign_15Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:15*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˛
save_1/Assign_16Assignpi/dense_2/biassave_1/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_1/Assign_17Assignpi/dense_2/bias/Adamsave_1/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_1/Assign_18Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_1/Assign_19Assignpi/dense_2/kernelsave_1/RestoreV2:19*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ŕ
save_1/Assign_20Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_1/Assign_21Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:21*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
¨
save_1/Assign_22Assign
pi/log_stdsave_1/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
­
save_1/Assign_23Assignpi/log_std/Adamsave_1/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_1/Assign_24Assignpi/log_std/Adam_1save_1/RestoreV2:24*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
­
save_1/Assign_25Assignv/dense/biassave_1/RestoreV2:25*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
˛
save_1/Assign_26Assignv/dense/bias/Adamsave_1/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_1/Assign_27Assignv/dense/bias/Adam_1save_1/RestoreV2:27*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
ľ
save_1/Assign_28Assignv/dense/kernelsave_1/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ş
save_1/Assign_29Assignv/dense/kernel/Adamsave_1/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_1/Assign_30Assignv/dense/kernel/Adam_1save_1/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ą
save_1/Assign_31Assignv/dense_1/biassave_1/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ś
save_1/Assign_32Assignv/dense_1/bias/Adamsave_1/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_1/Assign_33Assignv/dense_1/bias/Adam_1save_1/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ş
save_1/Assign_34Assignv/dense_1/kernelsave_1/RestoreV2:34*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
ż
save_1/Assign_35Assignv/dense_1/kernel/Adamsave_1/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_1/Assign_36Assignv/dense_1/kernel/Adam_1save_1/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

°
save_1/Assign_37Assignv/dense_2/biassave_1/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ľ
save_1/Assign_38Assignv/dense_2/bias/Adamsave_1/RestoreV2:38*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_1/Assign_39Assignv/dense_2/bias/Adam_1save_1/RestoreV2:39*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_1/Assign_40Assignv/dense_2/kernelsave_1/RestoreV2:40*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
ž
save_1/Assign_41Assignv/dense_2/kernel/Adamsave_1/RestoreV2:41*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Ŕ
save_1/Assign_42Assignv/dense_2/kernel/Adam_1save_1/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Á
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
_output_shapes
: *
shape: 

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_53ead0f7883f45299f4bc679f6be875b/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_2/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
ş
save_2/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ť
save_2/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ü
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
Ł
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
˝
save_2/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_2/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
í
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*9
dtypes/
-2+*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::
˘
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
§
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ś
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
§
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
­
save_2/Assign_4Assignpi/dense/biassave_2/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
˛
save_2/Assign_5Assignpi/dense/bias/Adamsave_2/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_2/Assign_6Assignpi/dense/bias/Adam_1save_2/RestoreV2:6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ľ
save_2/Assign_7Assignpi/dense/kernelsave_2/RestoreV2:7*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ş
save_2/Assign_8Assignpi/dense/kernel/Adamsave_2/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_2/Assign_9Assignpi/dense/kernel/Adam_1save_2/RestoreV2:9*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ł
save_2/Assign_10Assignpi/dense_1/biassave_2/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_2/Assign_11Assignpi/dense_1/bias/Adamsave_2/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_2/Assign_12Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_2/Assign_13Assignpi/dense_1/kernelsave_2/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_2/Assign_14Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_2/Assign_15Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:15*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˛
save_2/Assign_16Assignpi/dense_2/biassave_2/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_2/Assign_17Assignpi/dense_2/bias/Adamsave_2/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_2/Assign_18Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_2/Assign_19Assignpi/dense_2/kernelsave_2/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ŕ
save_2/Assign_20Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_2/Assign_21Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:21*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
¨
save_2/Assign_22Assign
pi/log_stdsave_2/RestoreV2:22*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
­
save_2/Assign_23Assignpi/log_std/Adamsave_2/RestoreV2:23*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_2/Assign_24Assignpi/log_std/Adam_1save_2/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
­
save_2/Assign_25Assignv/dense/biassave_2/RestoreV2:25*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
save_2/Assign_26Assignv/dense/bias/Adamsave_2/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_2/Assign_27Assignv/dense/bias/Adam_1save_2/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_2/Assign_28Assignv/dense/kernelsave_2/RestoreV2:28*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ş
save_2/Assign_29Assignv/dense/kernel/Adamsave_2/RestoreV2:29*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ź
save_2/Assign_30Assignv/dense/kernel/Adam_1save_2/RestoreV2:30*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ą
save_2/Assign_31Assignv/dense_1/biassave_2/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ś
save_2/Assign_32Assignv/dense_1/bias/Adamsave_2/RestoreV2:32*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_2/Assign_33Assignv/dense_1/bias/Adam_1save_2/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_2/Assign_34Assignv/dense_1/kernelsave_2/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save_2/Assign_35Assignv/dense_1/kernel/Adamsave_2/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Á
save_2/Assign_36Assignv/dense_1/kernel/Adam_1save_2/RestoreV2:36*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
°
save_2/Assign_37Assignv/dense_2/biassave_2/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ľ
save_2/Assign_38Assignv/dense_2/bias/Adamsave_2/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_2/Assign_39Assignv/dense_2/bias/Adam_1save_2/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
š
save_2/Assign_40Assignv/dense_2/kernelsave_2/RestoreV2:40*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
ž
save_2/Assign_41Assignv/dense_2/kernel/Adamsave_2/RestoreV2:41*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ŕ
save_2/Assign_42Assignv/dense_2/kernel/Adam_1save_2/RestoreV2:42*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Á
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
dtype0*
_output_shapes
: *
shape: 

save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_07c16b09dbd54f9bb7e29b1fbd23c8f8/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
ş
save_3/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ť
save_3/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ü
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_3/ShardedFilename
Ł
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(

save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
˝
save_3/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_3/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
í
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*9
dtypes/
-2+*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::
˘
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
§
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ś
save_3/Assign_2Assignbeta2_powersave_3/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
§
save_3/Assign_3Assignbeta2_power_1save_3/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
­
save_3/Assign_4Assignpi/dense/biassave_3/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
save_3/Assign_5Assignpi/dense/bias/Adamsave_3/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_3/Assign_6Assignpi/dense/bias/Adam_1save_3/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_3/Assign_7Assignpi/dense/kernelsave_3/RestoreV2:7*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ş
save_3/Assign_8Assignpi/dense/kernel/Adamsave_3/RestoreV2:8*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ź
save_3/Assign_9Assignpi/dense/kernel/Adam_1save_3/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_3/Assign_10Assignpi/dense_1/biassave_3/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_3/Assign_11Assignpi/dense_1/bias/Adamsave_3/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_3/Assign_12Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_3/Assign_13Assignpi/dense_1/kernelsave_3/RestoreV2:13*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Á
save_3/Assign_14Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_3/Assign_15Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:15*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˛
save_3/Assign_16Assignpi/dense_2/biassave_3/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_3/Assign_17Assignpi/dense_2/bias/Adamsave_3/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_3/Assign_18Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_3/Assign_19Assignpi/dense_2/kernelsave_3/RestoreV2:19*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ŕ
save_3/Assign_20Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:20*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Â
save_3/Assign_21Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:21*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
¨
save_3/Assign_22Assign
pi/log_stdsave_3/RestoreV2:22*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
­
save_3/Assign_23Assignpi/log_std/Adamsave_3/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
Ż
save_3/Assign_24Assignpi/log_std/Adam_1save_3/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
­
save_3/Assign_25Assignv/dense/biassave_3/RestoreV2:25*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
save_3/Assign_26Assignv/dense/bias/Adamsave_3/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_3/Assign_27Assignv/dense/bias/Adam_1save_3/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_3/Assign_28Assignv/dense/kernelsave_3/RestoreV2:28*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ş
save_3/Assign_29Assignv/dense/kernel/Adamsave_3/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_3/Assign_30Assignv/dense/kernel/Adam_1save_3/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ą
save_3/Assign_31Assignv/dense_1/biassave_3/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ś
save_3/Assign_32Assignv/dense_1/bias/Adamsave_3/RestoreV2:32*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_3/Assign_33Assignv/dense_1/bias/Adam_1save_3/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_3/Assign_34Assignv/dense_1/kernelsave_3/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save_3/Assign_35Assignv/dense_1/kernel/Adamsave_3/RestoreV2:35*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
Á
save_3/Assign_36Assignv/dense_1/kernel/Adam_1save_3/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

°
save_3/Assign_37Assignv/dense_2/biassave_3/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ľ
save_3/Assign_38Assignv/dense_2/bias/Adamsave_3/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_3/Assign_39Assignv/dense_2/bias/Adam_1save_3/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_3/Assign_40Assignv/dense_2/kernelsave_3/RestoreV2:40*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
ž
save_3/Assign_41Assignv/dense_2/kernel/Adamsave_3/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_3/Assign_42Assignv/dense_2/kernel/Adam_1save_3/RestoreV2:42*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Á
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
dtype0*
_output_shapes
: *
shape: 

save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_05fa2dcc9fc44a2dbb94b9e3fb4b817e/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_4/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
ş
save_4/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ť
save_4/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ü
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
Ł
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(

save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
_output_shapes
: *
T0
˝
save_4/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_4/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
í
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
˘
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
§
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
Ś
save_4/Assign_2Assignbeta2_powersave_4/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
§
save_4/Assign_3Assignbeta2_power_1save_4/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
­
save_4/Assign_4Assignpi/dense/biassave_4/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
˛
save_4/Assign_5Assignpi/dense/bias/Adamsave_4/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_4/Assign_6Assignpi/dense/bias/Adam_1save_4/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_4/Assign_7Assignpi/dense/kernelsave_4/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ş
save_4/Assign_8Assignpi/dense/kernel/Adamsave_4/RestoreV2:8*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ź
save_4/Assign_9Assignpi/dense/kernel/Adam_1save_4/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_4/Assign_10Assignpi/dense_1/biassave_4/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_4/Assign_11Assignpi/dense_1/bias/Adamsave_4/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_4/Assign_12Assignpi/dense_1/bias/Adam_1save_4/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_4/Assign_13Assignpi/dense_1/kernelsave_4/RestoreV2:13*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Á
save_4/Assign_14Assignpi/dense_1/kernel/Adamsave_4/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_4/Assign_15Assignpi/dense_1/kernel/Adam_1save_4/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_4/Assign_16Assignpi/dense_2/biassave_4/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_4/Assign_17Assignpi/dense_2/bias/Adamsave_4/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_4/Assign_18Assignpi/dense_2/bias/Adam_1save_4/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_4/Assign_19Assignpi/dense_2/kernelsave_4/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ŕ
save_4/Assign_20Assignpi/dense_2/kernel/Adamsave_4/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_4/Assign_21Assignpi/dense_2/kernel/Adam_1save_4/RestoreV2:21*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
¨
save_4/Assign_22Assign
pi/log_stdsave_4/RestoreV2:22*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
­
save_4/Assign_23Assignpi/log_std/Adamsave_4/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
Ż
save_4/Assign_24Assignpi/log_std/Adam_1save_4/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
­
save_4/Assign_25Assignv/dense/biassave_4/RestoreV2:25*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
save_4/Assign_26Assignv/dense/bias/Adamsave_4/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_4/Assign_27Assignv/dense/bias/Adam_1save_4/RestoreV2:27*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ľ
save_4/Assign_28Assignv/dense/kernelsave_4/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ş
save_4/Assign_29Assignv/dense/kernel/Adamsave_4/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_4/Assign_30Assignv/dense/kernel/Adam_1save_4/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ą
save_4/Assign_31Assignv/dense_1/biassave_4/RestoreV2:31*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_4/Assign_32Assignv/dense_1/bias/Adamsave_4/RestoreV2:32*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_4/Assign_33Assignv/dense_1/bias/Adam_1save_4/RestoreV2:33*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_4/Assign_34Assignv/dense_1/kernelsave_4/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save_4/Assign_35Assignv/dense_1/kernel/Adamsave_4/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Á
save_4/Assign_36Assignv/dense_1/kernel/Adam_1save_4/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

°
save_4/Assign_37Assignv/dense_2/biassave_4/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ľ
save_4/Assign_38Assignv/dense_2/bias/Adamsave_4/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_4/Assign_39Assignv/dense_2/bias/Adam_1save_4/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_4/Assign_40Assignv/dense_2/kernelsave_4/RestoreV2:40*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
ž
save_4/Assign_41Assignv/dense_2/kernel/Adamsave_4/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_4/Assign_42Assignv/dense_2/kernel/Adam_1save_4/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Á
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
dtype0*
_output_shapes
: *
shape: 

save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_718f8de4a5db47ca975bbb83984b523c/part*
dtype0*
_output_shapes
: 
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
ş
save_5/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:+*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
ť
save_5/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ü
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: 
Ł
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(

save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
˝
save_5/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_5/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
í
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
˘
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
§
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ś
save_5/Assign_2Assignbeta2_powersave_5/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
§
save_5/Assign_3Assignbeta2_power_1save_5/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
­
save_5/Assign_4Assignpi/dense/biassave_5/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
˛
save_5/Assign_5Assignpi/dense/bias/Adamsave_5/RestoreV2:5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
´
save_5/Assign_6Assignpi/dense/bias/Adam_1save_5/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ľ
save_5/Assign_7Assignpi/dense/kernelsave_5/RestoreV2:7*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ş
save_5/Assign_8Assignpi/dense/kernel/Adamsave_5/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_5/Assign_9Assignpi/dense/kernel/Adam_1save_5/RestoreV2:9*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ł
save_5/Assign_10Assignpi/dense_1/biassave_5/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_5/Assign_11Assignpi/dense_1/bias/Adamsave_5/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_5/Assign_12Assignpi/dense_1/bias/Adam_1save_5/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_5/Assign_13Assignpi/dense_1/kernelsave_5/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_5/Assign_14Assignpi/dense_1/kernel/Adamsave_5/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_5/Assign_15Assignpi/dense_1/kernel/Adam_1save_5/RestoreV2:15*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
˛
save_5/Assign_16Assignpi/dense_2/biassave_5/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
ˇ
save_5/Assign_17Assignpi/dense_2/bias/Adamsave_5/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_5/Assign_18Assignpi/dense_2/bias/Adam_1save_5/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_5/Assign_19Assignpi/dense_2/kernelsave_5/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_5/Assign_20Assignpi/dense_2/kernel/Adamsave_5/RestoreV2:20*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Â
save_5/Assign_21Assignpi/dense_2/kernel/Adam_1save_5/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
¨
save_5/Assign_22Assign
pi/log_stdsave_5/RestoreV2:22*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
­
save_5/Assign_23Assignpi/log_std/Adamsave_5/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_5/Assign_24Assignpi/log_std/Adam_1save_5/RestoreV2:24*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
­
save_5/Assign_25Assignv/dense/biassave_5/RestoreV2:25*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
save_5/Assign_26Assignv/dense/bias/Adamsave_5/RestoreV2:26*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_5/Assign_27Assignv/dense/bias/Adam_1save_5/RestoreV2:27*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ľ
save_5/Assign_28Assignv/dense/kernelsave_5/RestoreV2:28*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ş
save_5/Assign_29Assignv/dense/kernel/Adamsave_5/RestoreV2:29*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ź
save_5/Assign_30Assignv/dense/kernel/Adam_1save_5/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ą
save_5/Assign_31Assignv/dense_1/biassave_5/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ś
save_5/Assign_32Assignv/dense_1/bias/Adamsave_5/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_5/Assign_33Assignv/dense_1/bias/Adam_1save_5/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_5/Assign_34Assignv/dense_1/kernelsave_5/RestoreV2:34*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
ż
save_5/Assign_35Assignv/dense_1/kernel/Adamsave_5/RestoreV2:35*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
Á
save_5/Assign_36Assignv/dense_1/kernel/Adam_1save_5/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

°
save_5/Assign_37Assignv/dense_2/biassave_5/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ľ
save_5/Assign_38Assignv/dense_2/bias/Adamsave_5/RestoreV2:38*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ˇ
save_5/Assign_39Assignv/dense_2/bias/Adam_1save_5/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
š
save_5/Assign_40Assignv/dense_2/kernelsave_5/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ž
save_5/Assign_41Assignv/dense_2/kernel/Adamsave_5/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_5/Assign_42Assignv/dense_2/kernel/Adam_1save_5/RestoreV2:42*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Á
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
dtype0*
_output_shapes
: *
shape: 

save_6/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_51e7066be7ff44c9a44daaff4cadef19/part
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
ş
save_6/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ť
save_6/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ü
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: 
Ł
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(

save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
T0*
_output_shapes
: 
˝
save_6/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_6/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
í
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*9
dtypes/
-2+*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::
˘
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
§
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ś
save_6/Assign_2Assignbeta2_powersave_6/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
§
save_6/Assign_3Assignbeta2_power_1save_6/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
­
save_6/Assign_4Assignpi/dense/biassave_6/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
˛
save_6/Assign_5Assignpi/dense/bias/Adamsave_6/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_6/Assign_6Assignpi/dense/bias/Adam_1save_6/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_6/Assign_7Assignpi/dense/kernelsave_6/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ş
save_6/Assign_8Assignpi/dense/kernel/Adamsave_6/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_6/Assign_9Assignpi/dense/kernel/Adam_1save_6/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_6/Assign_10Assignpi/dense_1/biassave_6/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
¸
save_6/Assign_11Assignpi/dense_1/bias/Adamsave_6/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_6/Assign_12Assignpi/dense_1/bias/Adam_1save_6/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_6/Assign_13Assignpi/dense_1/kernelsave_6/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_6/Assign_14Assignpi/dense_1/kernel/Adamsave_6/RestoreV2:14*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Ă
save_6/Assign_15Assignpi/dense_1/kernel/Adam_1save_6/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_6/Assign_16Assignpi/dense_2/biassave_6/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_6/Assign_17Assignpi/dense_2/bias/Adamsave_6/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_6/Assign_18Assignpi/dense_2/bias/Adam_1save_6/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_6/Assign_19Assignpi/dense_2/kernelsave_6/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_6/Assign_20Assignpi/dense_2/kernel/Adamsave_6/RestoreV2:20*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Â
save_6/Assign_21Assignpi/dense_2/kernel/Adam_1save_6/RestoreV2:21*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
¨
save_6/Assign_22Assign
pi/log_stdsave_6/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
­
save_6/Assign_23Assignpi/log_std/Adamsave_6/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_6/Assign_24Assignpi/log_std/Adam_1save_6/RestoreV2:24*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
­
save_6/Assign_25Assignv/dense/biassave_6/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
˛
save_6/Assign_26Assignv/dense/bias/Adamsave_6/RestoreV2:26*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_6/Assign_27Assignv/dense/bias/Adam_1save_6/RestoreV2:27*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ľ
save_6/Assign_28Assignv/dense/kernelsave_6/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ş
save_6/Assign_29Assignv/dense/kernel/Adamsave_6/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_6/Assign_30Assignv/dense/kernel/Adam_1save_6/RestoreV2:30*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ą
save_6/Assign_31Assignv/dense_1/biassave_6/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ś
save_6/Assign_32Assignv/dense_1/bias/Adamsave_6/RestoreV2:32*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
¸
save_6/Assign_33Assignv/dense_1/bias/Adam_1save_6/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_6/Assign_34Assignv/dense_1/kernelsave_6/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save_6/Assign_35Assignv/dense_1/kernel/Adamsave_6/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_6/Assign_36Assignv/dense_1/kernel/Adam_1save_6/RestoreV2:36*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
°
save_6/Assign_37Assignv/dense_2/biassave_6/RestoreV2:37*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ľ
save_6/Assign_38Assignv/dense_2/bias/Adamsave_6/RestoreV2:38*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_6/Assign_39Assignv/dense_2/bias/Adam_1save_6/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
š
save_6/Assign_40Assignv/dense_2/kernelsave_6/RestoreV2:40*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
ž
save_6/Assign_41Assignv/dense_2/kernel/Adamsave_6/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_6/Assign_42Assignv/dense_2/kernel/Adam_1save_6/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Á
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
shape: *
dtype0*
_output_shapes
: 

save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_dcf538d3d39e49e9940e21ed4a0c67a3/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_7/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
ş
save_7/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ť
save_7/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ü
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
T0*)
_class
loc:@save_7/ShardedFilename*
_output_shapes
: 
Ł
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(

save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
˝
save_7/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_7/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
í
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
˘
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
§
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ś
save_7/Assign_2Assignbeta2_powersave_7/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
§
save_7/Assign_3Assignbeta2_power_1save_7/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
­
save_7/Assign_4Assignpi/dense/biassave_7/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
˛
save_7/Assign_5Assignpi/dense/bias/Adamsave_7/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_7/Assign_6Assignpi/dense/bias/Adam_1save_7/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_7/Assign_7Assignpi/dense/kernelsave_7/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ş
save_7/Assign_8Assignpi/dense/kernel/Adamsave_7/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_7/Assign_9Assignpi/dense/kernel/Adam_1save_7/RestoreV2:9*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ł
save_7/Assign_10Assignpi/dense_1/biassave_7/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_7/Assign_11Assignpi/dense_1/bias/Adamsave_7/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ş
save_7/Assign_12Assignpi/dense_1/bias/Adam_1save_7/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_7/Assign_13Assignpi/dense_1/kernelsave_7/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_7/Assign_14Assignpi/dense_1/kernel/Adamsave_7/RestoreV2:14*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Ă
save_7/Assign_15Assignpi/dense_1/kernel/Adam_1save_7/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_7/Assign_16Assignpi/dense_2/biassave_7/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_7/Assign_17Assignpi/dense_2/bias/Adamsave_7/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_7/Assign_18Assignpi/dense_2/bias/Adam_1save_7/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_7/Assign_19Assignpi/dense_2/kernelsave_7/RestoreV2:19*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ŕ
save_7/Assign_20Assignpi/dense_2/kernel/Adamsave_7/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_7/Assign_21Assignpi/dense_2/kernel/Adam_1save_7/RestoreV2:21*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
¨
save_7/Assign_22Assign
pi/log_stdsave_7/RestoreV2:22*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
­
save_7/Assign_23Assignpi/log_std/Adamsave_7/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_7/Assign_24Assignpi/log_std/Adam_1save_7/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
­
save_7/Assign_25Assignv/dense/biassave_7/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
˛
save_7/Assign_26Assignv/dense/bias/Adamsave_7/RestoreV2:26*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
´
save_7/Assign_27Assignv/dense/bias/Adam_1save_7/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_7/Assign_28Assignv/dense/kernelsave_7/RestoreV2:28*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ş
save_7/Assign_29Assignv/dense/kernel/Adamsave_7/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_7/Assign_30Assignv/dense/kernel/Adam_1save_7/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ą
save_7/Assign_31Assignv/dense_1/biassave_7/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ś
save_7/Assign_32Assignv/dense_1/bias/Adamsave_7/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_7/Assign_33Assignv/dense_1/bias/Adam_1save_7/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ş
save_7/Assign_34Assignv/dense_1/kernelsave_7/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save_7/Assign_35Assignv/dense_1/kernel/Adamsave_7/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_7/Assign_36Assignv/dense_1/kernel/Adam_1save_7/RestoreV2:36*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
°
save_7/Assign_37Assignv/dense_2/biassave_7/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ľ
save_7/Assign_38Assignv/dense_2/bias/Adamsave_7/RestoreV2:38*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ˇ
save_7/Assign_39Assignv/dense_2/bias/Adam_1save_7/RestoreV2:39*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_7/Assign_40Assignv/dense_2/kernelsave_7/RestoreV2:40*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
ž
save_7/Assign_41Assignv/dense_2/kernel/Adamsave_7/RestoreV2:41*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Ŕ
save_7/Assign_42Assignv/dense_2/kernel/Adam_1save_7/RestoreV2:42*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Á
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
shape: *
dtype0*
_output_shapes
: 

save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_0f40ff837d314abda29dd7e7e8bdb841/part*
dtype0*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_8/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_8/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
ş
save_8/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ť
save_8/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ü
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
T0*)
_class
loc:@save_8/ShardedFilename*
_output_shapes
: 
Ł
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(

save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
˝
save_8/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_8/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
í
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
˘
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
§
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ś
save_8/Assign_2Assignbeta2_powersave_8/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
§
save_8/Assign_3Assignbeta2_power_1save_8/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
­
save_8/Assign_4Assignpi/dense/biassave_8/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
save_8/Assign_5Assignpi/dense/bias/Adamsave_8/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_8/Assign_6Assignpi/dense/bias/Adam_1save_8/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_8/Assign_7Assignpi/dense/kernelsave_8/RestoreV2:7*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ş
save_8/Assign_8Assignpi/dense/kernel/Adamsave_8/RestoreV2:8*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ź
save_8/Assign_9Assignpi/dense/kernel/Adam_1save_8/RestoreV2:9*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ł
save_8/Assign_10Assignpi/dense_1/biassave_8/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_8/Assign_11Assignpi/dense_1/bias/Adamsave_8/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_8/Assign_12Assignpi/dense_1/bias/Adam_1save_8/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_8/Assign_13Assignpi/dense_1/kernelsave_8/RestoreV2:13*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Á
save_8/Assign_14Assignpi/dense_1/kernel/Adamsave_8/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_8/Assign_15Assignpi/dense_1/kernel/Adam_1save_8/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_8/Assign_16Assignpi/dense_2/biassave_8/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_8/Assign_17Assignpi/dense_2/bias/Adamsave_8/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_8/Assign_18Assignpi/dense_2/bias/Adam_1save_8/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_8/Assign_19Assignpi/dense_2/kernelsave_8/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_8/Assign_20Assignpi/dense_2/kernel/Adamsave_8/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_8/Assign_21Assignpi/dense_2/kernel/Adam_1save_8/RestoreV2:21*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
¨
save_8/Assign_22Assign
pi/log_stdsave_8/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
­
save_8/Assign_23Assignpi/log_std/Adamsave_8/RestoreV2:23*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_8/Assign_24Assignpi/log_std/Adam_1save_8/RestoreV2:24*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
­
save_8/Assign_25Assignv/dense/biassave_8/RestoreV2:25*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
save_8/Assign_26Assignv/dense/bias/Adamsave_8/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_8/Assign_27Assignv/dense/bias/Adam_1save_8/RestoreV2:27*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
ľ
save_8/Assign_28Assignv/dense/kernelsave_8/RestoreV2:28*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ş
save_8/Assign_29Assignv/dense/kernel/Adamsave_8/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_8/Assign_30Assignv/dense/kernel/Adam_1save_8/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ą
save_8/Assign_31Assignv/dense_1/biassave_8/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ś
save_8/Assign_32Assignv/dense_1/bias/Adamsave_8/RestoreV2:32*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_8/Assign_33Assignv/dense_1/bias/Adam_1save_8/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_8/Assign_34Assignv/dense_1/kernelsave_8/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

ż
save_8/Assign_35Assignv/dense_1/kernel/Adamsave_8/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Á
save_8/Assign_36Assignv/dense_1/kernel/Adam_1save_8/RestoreV2:36*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
°
save_8/Assign_37Assignv/dense_2/biassave_8/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ľ
save_8/Assign_38Assignv/dense_2/bias/Adamsave_8/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_8/Assign_39Assignv/dense_2/bias/Adam_1save_8/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_8/Assign_40Assignv/dense_2/kernelsave_8/RestoreV2:40*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
ž
save_8/Assign_41Assignv/dense_2/kernel/Adamsave_8/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_8/Assign_42Assignv/dense_2/kernel/Adam_1save_8/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Á
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
dtype0*
_output_shapes
: *
shape: 

save_9/StringJoin/inputs_1Const*<
value3B1 B+_temp_2c485d4b695c426096debaf92099f83e/part*
dtype0*
_output_shapes
: 
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_9/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_9/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
ş
save_9/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ť
save_9/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ü
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*)
_class
loc:@save_9/ShardedFilename*
_output_shapes
: 
Ł
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(

save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
˝
save_9/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ž
!save_9/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
í
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
˘
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
§
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
Ś
save_9/Assign_2Assignbeta2_powersave_9/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
§
save_9/Assign_3Assignbeta2_power_1save_9/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
­
save_9/Assign_4Assignpi/dense/biassave_9/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
˛
save_9/Assign_5Assignpi/dense/bias/Adamsave_9/RestoreV2:5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
´
save_9/Assign_6Assignpi/dense/bias/Adam_1save_9/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ľ
save_9/Assign_7Assignpi/dense/kernelsave_9/RestoreV2:7*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ş
save_9/Assign_8Assignpi/dense/kernel/Adamsave_9/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_9/Assign_9Assignpi/dense/kernel/Adam_1save_9/RestoreV2:9*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ł
save_9/Assign_10Assignpi/dense_1/biassave_9/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_9/Assign_11Assignpi/dense_1/bias/Adamsave_9/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_9/Assign_12Assignpi/dense_1/bias/Adam_1save_9/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_9/Assign_13Assignpi/dense_1/kernelsave_9/RestoreV2:13*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Á
save_9/Assign_14Assignpi/dense_1/kernel/Adamsave_9/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_9/Assign_15Assignpi/dense_1/kernel/Adam_1save_9/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_9/Assign_16Assignpi/dense_2/biassave_9/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
ˇ
save_9/Assign_17Assignpi/dense_2/bias/Adamsave_9/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_9/Assign_18Assignpi/dense_2/bias/Adam_1save_9/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_9/Assign_19Assignpi/dense_2/kernelsave_9/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ŕ
save_9/Assign_20Assignpi/dense_2/kernel/Adamsave_9/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_9/Assign_21Assignpi/dense_2/kernel/Adam_1save_9/RestoreV2:21*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
¨
save_9/Assign_22Assign
pi/log_stdsave_9/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
­
save_9/Assign_23Assignpi/log_std/Adamsave_9/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
Ż
save_9/Assign_24Assignpi/log_std/Adam_1save_9/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
­
save_9/Assign_25Assignv/dense/biassave_9/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
˛
save_9/Assign_26Assignv/dense/bias/Adamsave_9/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_9/Assign_27Assignv/dense/bias/Adam_1save_9/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ľ
save_9/Assign_28Assignv/dense/kernelsave_9/RestoreV2:28*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ş
save_9/Assign_29Assignv/dense/kernel/Adamsave_9/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_9/Assign_30Assignv/dense/kernel/Adam_1save_9/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ą
save_9/Assign_31Assignv/dense_1/biassave_9/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ś
save_9/Assign_32Assignv/dense_1/bias/Adamsave_9/RestoreV2:32*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
¸
save_9/Assign_33Assignv/dense_1/bias/Adam_1save_9/RestoreV2:33*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_9/Assign_34Assignv/dense_1/kernelsave_9/RestoreV2:34*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ż
save_9/Assign_35Assignv/dense_1/kernel/Adamsave_9/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_9/Assign_36Assignv/dense_1/kernel/Adam_1save_9/RestoreV2:36*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
°
save_9/Assign_37Assignv/dense_2/biassave_9/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ľ
save_9/Assign_38Assignv/dense_2/bias/Adamsave_9/RestoreV2:38*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_9/Assign_39Assignv/dense_2/bias/Adam_1save_9/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_9/Assign_40Assignv/dense_2/kernelsave_9/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
ž
save_9/Assign_41Assignv/dense_2/kernel/Adamsave_9/RestoreV2:41*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ŕ
save_9/Assign_42Assignv/dense_2/kernel/Adam_1save_9/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Á
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
dtype0*
_output_shapes
: *
shape: 

save_10/StringJoin/inputs_1Const*<
value3B1 B+_temp_9c3b37bb0f3a43f886c3d44170b24ecb/part*
dtype0*
_output_shapes
: 
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_10/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_10/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
ť
save_10/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_10/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
T0**
_class 
loc:@save_10/ShardedFilename*
_output_shapes
: 
Ś
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(

save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
T0*
_output_shapes
: 
ž
save_10/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_10/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ń
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Š
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
¨
save_10/Assign_2Assignbeta2_powersave_10/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Š
save_10/Assign_3Assignbeta2_power_1save_10/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ż
save_10/Assign_4Assignpi/dense/biassave_10/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_10/Assign_5Assignpi/dense/bias/Adamsave_10/RestoreV2:5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ś
save_10/Assign_6Assignpi/dense/bias/Adam_1save_10/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_10/Assign_7Assignpi/dense/kernelsave_10/RestoreV2:7*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_10/Assign_8Assignpi/dense/kernel/Adamsave_10/RestoreV2:8*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ž
save_10/Assign_9Assignpi/dense/kernel/Adam_1save_10/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ľ
save_10/Assign_10Assignpi/dense_1/biassave_10/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_10/Assign_11Assignpi/dense_1/bias/Adamsave_10/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_10/Assign_12Assignpi/dense_1/bias/Adam_1save_10/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ž
save_10/Assign_13Assignpi/dense_1/kernelsave_10/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_10/Assign_14Assignpi/dense_1/kernel/Adamsave_10/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ĺ
save_10/Assign_15Assignpi/dense_1/kernel/Adam_1save_10/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_10/Assign_16Assignpi/dense_2/biassave_10/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_10/Assign_17Assignpi/dense_2/bias/Adamsave_10/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_10/Assign_18Assignpi/dense_2/bias/Adam_1save_10/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
˝
save_10/Assign_19Assignpi/dense_2/kernelsave_10/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_10/Assign_20Assignpi/dense_2/kernel/Adamsave_10/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_10/Assign_21Assignpi/dense_2/kernel/Adam_1save_10/RestoreV2:21*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ş
save_10/Assign_22Assign
pi/log_stdsave_10/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
Ż
save_10/Assign_23Assignpi/log_std/Adamsave_10/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
ą
save_10/Assign_24Assignpi/log_std/Adam_1save_10/RestoreV2:24*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_10/Assign_25Assignv/dense/biassave_10/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_10/Assign_26Assignv/dense/bias/Adamsave_10/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_10/Assign_27Assignv/dense/bias/Adam_1save_10/RestoreV2:27*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
ˇ
save_10/Assign_28Assignv/dense/kernelsave_10/RestoreV2:28*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_10/Assign_29Assignv/dense/kernel/Adamsave_10/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ž
save_10/Assign_30Assignv/dense/kernel/Adam_1save_10/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_10/Assign_31Assignv/dense_1/biassave_10/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
¸
save_10/Assign_32Assignv/dense_1/bias/Adamsave_10/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_10/Assign_33Assignv/dense_1/bias/Adam_1save_10/RestoreV2:33*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_10/Assign_34Assignv/dense_1/kernelsave_10/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_10/Assign_35Assignv/dense_1/kernel/Adamsave_10/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_10/Assign_36Assignv/dense_1/kernel/Adam_1save_10/RestoreV2:36*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
˛
save_10/Assign_37Assignv/dense_2/biassave_10/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_10/Assign_38Assignv/dense_2/bias/Adamsave_10/RestoreV2:38*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_10/Assign_39Assignv/dense_2/bias/Adam_1save_10/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ť
save_10/Assign_40Assignv/dense_2/kernelsave_10/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_10/Assign_41Assignv/dense_2/kernel/Adamsave_10/RestoreV2:41*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_10/Assign_42Assignv/dense_2/kernel/Adam_1save_10/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
í
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
dtype0*
_output_shapes
: *
shape: 

save_11/StringJoin/inputs_1Const*<
value3B1 B+_temp_a7fdda55668f47098efe5ff7af44bf65/part*
dtype0*
_output_shapes
: 
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_11/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_11/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
ť
save_11/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_11/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
T0**
_class 
loc:@save_11/ShardedFilename*
_output_shapes
: 
Ś
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(

save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
T0*
_output_shapes
: 
ž
save_11/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_11/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ń
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_11/AssignAssignbeta1_powersave_11/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Š
save_11/Assign_1Assignbeta1_power_1save_11/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
¨
save_11/Assign_2Assignbeta2_powersave_11/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Š
save_11/Assign_3Assignbeta2_power_1save_11/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ż
save_11/Assign_4Assignpi/dense/biassave_11/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_11/Assign_5Assignpi/dense/bias/Adamsave_11/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_11/Assign_6Assignpi/dense/bias/Adam_1save_11/RestoreV2:6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ˇ
save_11/Assign_7Assignpi/dense/kernelsave_11/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_11/Assign_8Assignpi/dense/kernel/Adamsave_11/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_11/Assign_9Assignpi/dense/kernel/Adam_1save_11/RestoreV2:9*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ľ
save_11/Assign_10Assignpi/dense_1/biassave_11/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_11/Assign_11Assignpi/dense_1/bias/Adamsave_11/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_11/Assign_12Assignpi/dense_1/bias/Adam_1save_11/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ž
save_11/Assign_13Assignpi/dense_1/kernelsave_11/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_11/Assign_14Assignpi/dense_1/kernel/Adamsave_11/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ĺ
save_11/Assign_15Assignpi/dense_1/kernel/Adam_1save_11/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_11/Assign_16Assignpi/dense_2/biassave_11/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_11/Assign_17Assignpi/dense_2/bias/Adamsave_11/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_11/Assign_18Assignpi/dense_2/bias/Adam_1save_11/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
˝
save_11/Assign_19Assignpi/dense_2/kernelsave_11/RestoreV2:19*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Â
save_11/Assign_20Assignpi/dense_2/kernel/Adamsave_11/RestoreV2:20*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ä
save_11/Assign_21Assignpi/dense_2/kernel/Adam_1save_11/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ş
save_11/Assign_22Assign
pi/log_stdsave_11/RestoreV2:22*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_11/Assign_23Assignpi/log_std/Adamsave_11/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
ą
save_11/Assign_24Assignpi/log_std/Adam_1save_11/RestoreV2:24*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
Ż
save_11/Assign_25Assignv/dense/biassave_11/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_11/Assign_26Assignv/dense/bias/Adamsave_11/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_11/Assign_27Assignv/dense/bias/Adam_1save_11/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_11/Assign_28Assignv/dense/kernelsave_11/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_11/Assign_29Assignv/dense/kernel/Adamsave_11/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_11/Assign_30Assignv/dense/kernel/Adam_1save_11/RestoreV2:30*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ł
save_11/Assign_31Assignv/dense_1/biassave_11/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_11/Assign_32Assignv/dense_1/bias/Adamsave_11/RestoreV2:32*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_11/Assign_33Assignv/dense_1/bias/Adam_1save_11/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_11/Assign_34Assignv/dense_1/kernelsave_11/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_11/Assign_35Assignv/dense_1/kernel/Adamsave_11/RestoreV2:35*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
Ă
save_11/Assign_36Assignv/dense_1/kernel/Adam_1save_11/RestoreV2:36*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_11/Assign_37Assignv/dense_2/biassave_11/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_11/Assign_38Assignv/dense_2/bias/Adamsave_11/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_11/Assign_39Assignv/dense_2/bias/Adam_1save_11/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_11/Assign_40Assignv/dense_2/kernelsave_11/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_11/Assign_41Assignv/dense_2/kernel/Adamsave_11/RestoreV2:41*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_11/Assign_42Assignv/dense_2/kernel/Adam_1save_11/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
í
save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_38^save_11/Assign_39^save_11/Assign_4^save_11/Assign_40^save_11/Assign_41^save_11/Assign_42^save_11/Assign_5^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
dtype0*
_output_shapes
: *
shape: 

save_12/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_77d5a57a8c694195b952e37c45b40b42/part
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_12/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_12/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
ť
save_12/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_12/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
T0**
_class 
loc:@save_12/ShardedFilename*
_output_shapes
: 
Ś
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(

save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
T0*
_output_shapes
: 
ž
save_12/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:+*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
ż
"save_12/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ń
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Š
save_12/Assign_1Assignbeta1_power_1save_12/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
¨
save_12/Assign_2Assignbeta2_powersave_12/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Š
save_12/Assign_3Assignbeta2_power_1save_12/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ż
save_12/Assign_4Assignpi/dense/biassave_12/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_12/Assign_5Assignpi/dense/bias/Adamsave_12/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_12/Assign_6Assignpi/dense/bias/Adam_1save_12/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_12/Assign_7Assignpi/dense/kernelsave_12/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_12/Assign_8Assignpi/dense/kernel/Adamsave_12/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_12/Assign_9Assignpi/dense/kernel/Adam_1save_12/RestoreV2:9*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ľ
save_12/Assign_10Assignpi/dense_1/biassave_12/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_12/Assign_11Assignpi/dense_1/bias/Adamsave_12/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_12/Assign_12Assignpi/dense_1/bias/Adam_1save_12/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ž
save_12/Assign_13Assignpi/dense_1/kernelsave_12/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_12/Assign_14Assignpi/dense_1/kernel/Adamsave_12/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ĺ
save_12/Assign_15Assignpi/dense_1/kernel/Adam_1save_12/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_12/Assign_16Assignpi/dense_2/biassave_12/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_12/Assign_17Assignpi/dense_2/bias/Adamsave_12/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_12/Assign_18Assignpi/dense_2/bias/Adam_1save_12/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
˝
save_12/Assign_19Assignpi/dense_2/kernelsave_12/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_12/Assign_20Assignpi/dense_2/kernel/Adamsave_12/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_12/Assign_21Assignpi/dense_2/kernel/Adam_1save_12/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ş
save_12/Assign_22Assign
pi/log_stdsave_12/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
Ż
save_12/Assign_23Assignpi/log_std/Adamsave_12/RestoreV2:23*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
ą
save_12/Assign_24Assignpi/log_std/Adam_1save_12/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_12/Assign_25Assignv/dense/biassave_12/RestoreV2:25*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_12/Assign_26Assignv/dense/bias/Adamsave_12/RestoreV2:26*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
ś
save_12/Assign_27Assignv/dense/bias/Adam_1save_12/RestoreV2:27*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ˇ
save_12/Assign_28Assignv/dense/kernelsave_12/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_12/Assign_29Assignv/dense/kernel/Adamsave_12/RestoreV2:29*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ž
save_12/Assign_30Assignv/dense/kernel/Adam_1save_12/RestoreV2:30*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ł
save_12/Assign_31Assignv/dense_1/biassave_12/RestoreV2:31*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_12/Assign_32Assignv/dense_1/bias/Adamsave_12/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_12/Assign_33Assignv/dense_1/bias/Adam_1save_12/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_12/Assign_34Assignv/dense_1/kernelsave_12/RestoreV2:34*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
Á
save_12/Assign_35Assignv/dense_1/kernel/Adamsave_12/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_12/Assign_36Assignv/dense_1/kernel/Adam_1save_12/RestoreV2:36*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_12/Assign_37Assignv/dense_2/biassave_12/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_12/Assign_38Assignv/dense_2/bias/Adamsave_12/RestoreV2:38*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
š
save_12/Assign_39Assignv/dense_2/bias/Adam_1save_12/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_12/Assign_40Assignv/dense_2/kernelsave_12/RestoreV2:40*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ŕ
save_12/Assign_41Assignv/dense_2/kernel/Adamsave_12/RestoreV2:41*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_12/Assign_42Assignv/dense_2/kernel/Adam_1save_12/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
í
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_40^save_12/Assign_41^save_12/Assign_42^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
dtype0*
_output_shapes
: *
shape: 

save_13/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_6c78bcd903564657ad128be0d5f8c250/part
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_13/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_13/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
ť
save_13/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_13/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2*
T0**
_class 
loc:@save_13/ShardedFilename*
_output_shapes
: 
Ś
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(

save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
T0*
_output_shapes
: 
ž
save_13/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_13/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ń
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Š
save_13/Assign_1Assignbeta1_power_1save_13/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
¨
save_13/Assign_2Assignbeta2_powersave_13/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Š
save_13/Assign_3Assignbeta2_power_1save_13/RestoreV2:3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
Ż
save_13/Assign_4Assignpi/dense/biassave_13/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_13/Assign_5Assignpi/dense/bias/Adamsave_13/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_13/Assign_6Assignpi/dense/bias/Adam_1save_13/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_13/Assign_7Assignpi/dense/kernelsave_13/RestoreV2:7*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ź
save_13/Assign_8Assignpi/dense/kernel/Adamsave_13/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_13/Assign_9Assignpi/dense/kernel/Adam_1save_13/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ľ
save_13/Assign_10Assignpi/dense_1/biassave_13/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ş
save_13/Assign_11Assignpi/dense_1/bias/Adamsave_13/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_13/Assign_12Assignpi/dense_1/bias/Adam_1save_13/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ž
save_13/Assign_13Assignpi/dense_1/kernelsave_13/RestoreV2:13*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Ă
save_13/Assign_14Assignpi/dense_1/kernel/Adamsave_13/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ĺ
save_13/Assign_15Assignpi/dense_1/kernel/Adam_1save_13/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
´
save_13/Assign_16Assignpi/dense_2/biassave_13/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_13/Assign_17Assignpi/dense_2/bias/Adamsave_13/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_13/Assign_18Assignpi/dense_2/bias/Adam_1save_13/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
˝
save_13/Assign_19Assignpi/dense_2/kernelsave_13/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_13/Assign_20Assignpi/dense_2/kernel/Adamsave_13/RestoreV2:20*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ä
save_13/Assign_21Assignpi/dense_2/kernel/Adam_1save_13/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ş
save_13/Assign_22Assign
pi/log_stdsave_13/RestoreV2:22*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_13/Assign_23Assignpi/log_std/Adamsave_13/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
ą
save_13/Assign_24Assignpi/log_std/Adam_1save_13/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_13/Assign_25Assignv/dense/biassave_13/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_13/Assign_26Assignv/dense/bias/Adamsave_13/RestoreV2:26*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_13/Assign_27Assignv/dense/bias/Adam_1save_13/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_13/Assign_28Assignv/dense/kernelsave_13/RestoreV2:28*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ź
save_13/Assign_29Assignv/dense/kernel/Adamsave_13/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_13/Assign_30Assignv/dense/kernel/Adam_1save_13/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_13/Assign_31Assignv/dense_1/biassave_13/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
¸
save_13/Assign_32Assignv/dense_1/bias/Adamsave_13/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_13/Assign_33Assignv/dense_1/bias/Adam_1save_13/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_13/Assign_34Assignv/dense_1/kernelsave_13/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_13/Assign_35Assignv/dense_1/kernel/Adamsave_13/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_13/Assign_36Assignv/dense_1/kernel/Adam_1save_13/RestoreV2:36*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_13/Assign_37Assignv/dense_2/biassave_13/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_13/Assign_38Assignv/dense_2/bias/Adamsave_13/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_13/Assign_39Assignv/dense_2/bias/Adam_1save_13/RestoreV2:39*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_13/Assign_40Assignv/dense_2/kernelsave_13/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_13/Assign_41Assignv/dense_2/kernel/Adamsave_13/RestoreV2:41*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_13/Assign_42Assignv/dense_2/kernel/Adam_1save_13/RestoreV2:42*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
í
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_40^save_13/Assign_41^save_13/Assign_42^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
dtype0*
_output_shapes
: *
shape: 

save_14/StringJoin/inputs_1Const*<
value3B1 B+_temp_3bf90e7fd99f415ebfb1b3d594c2bb99/part*
dtype0*
_output_shapes
: 
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_14/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_14/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
ť
save_14/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_14/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
T0**
_class 
loc:@save_14/ShardedFilename*
_output_shapes
: 
Ś
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(

save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
T0*
_output_shapes
: 
ž
save_14/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_14/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ń
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_14/AssignAssignbeta1_powersave_14/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Š
save_14/Assign_1Assignbeta1_power_1save_14/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
¨
save_14/Assign_2Assignbeta2_powersave_14/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Š
save_14/Assign_3Assignbeta2_power_1save_14/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ż
save_14/Assign_4Assignpi/dense/biassave_14/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_14/Assign_5Assignpi/dense/bias/Adamsave_14/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_14/Assign_6Assignpi/dense/bias/Adam_1save_14/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_14/Assign_7Assignpi/dense/kernelsave_14/RestoreV2:7*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_14/Assign_8Assignpi/dense/kernel/Adamsave_14/RestoreV2:8*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ž
save_14/Assign_9Assignpi/dense/kernel/Adam_1save_14/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ľ
save_14/Assign_10Assignpi/dense_1/biassave_14/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ş
save_14/Assign_11Assignpi/dense_1/bias/Adamsave_14/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_14/Assign_12Assignpi/dense_1/bias/Adam_1save_14/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ž
save_14/Assign_13Assignpi/dense_1/kernelsave_14/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_14/Assign_14Assignpi/dense_1/kernel/Adamsave_14/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ĺ
save_14/Assign_15Assignpi/dense_1/kernel/Adam_1save_14/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_14/Assign_16Assignpi/dense_2/biassave_14/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_14/Assign_17Assignpi/dense_2/bias/Adamsave_14/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_14/Assign_18Assignpi/dense_2/bias/Adam_1save_14/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
˝
save_14/Assign_19Assignpi/dense_2/kernelsave_14/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_14/Assign_20Assignpi/dense_2/kernel/Adamsave_14/RestoreV2:20*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ä
save_14/Assign_21Assignpi/dense_2/kernel/Adam_1save_14/RestoreV2:21*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ş
save_14/Assign_22Assign
pi/log_stdsave_14/RestoreV2:22*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_14/Assign_23Assignpi/log_std/Adamsave_14/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
ą
save_14/Assign_24Assignpi/log_std/Adam_1save_14/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_14/Assign_25Assignv/dense/biassave_14/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_14/Assign_26Assignv/dense/bias/Adamsave_14/RestoreV2:26*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
ś
save_14/Assign_27Assignv/dense/bias/Adam_1save_14/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_14/Assign_28Assignv/dense/kernelsave_14/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_14/Assign_29Assignv/dense/kernel/Adamsave_14/RestoreV2:29*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ž
save_14/Assign_30Assignv/dense/kernel/Adam_1save_14/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_14/Assign_31Assignv/dense_1/biassave_14/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
¸
save_14/Assign_32Assignv/dense_1/bias/Adamsave_14/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_14/Assign_33Assignv/dense_1/bias/Adam_1save_14/RestoreV2:33*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_14/Assign_34Assignv/dense_1/kernelsave_14/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_14/Assign_35Assignv/dense_1/kernel/Adamsave_14/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_14/Assign_36Assignv/dense_1/kernel/Adam_1save_14/RestoreV2:36*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_14/Assign_37Assignv/dense_2/biassave_14/RestoreV2:37*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ˇ
save_14/Assign_38Assignv/dense_2/bias/Adamsave_14/RestoreV2:38*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_14/Assign_39Assignv/dense_2/bias/Adam_1save_14/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ť
save_14/Assign_40Assignv/dense_2/kernelsave_14/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_14/Assign_41Assignv/dense_2/kernel/Adamsave_14/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_14/Assign_42Assignv/dense_2/kernel/Adam_1save_14/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
í
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_40^save_14/Assign_41^save_14/Assign_42^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
dtype0*
_output_shapes
: *
shape: 

save_15/StringJoin/inputs_1Const*<
value3B1 B+_temp_ff294200974e4ea6afaf7be79f2bd246/part*
dtype0*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_15/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
ť
save_15/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_15/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
T0**
_class 
loc:@save_15/ShardedFilename*
_output_shapes
: 
Ś
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(

save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
T0*
_output_shapes
: 
ž
save_15/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_15/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ń
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Š
save_15/Assign_1Assignbeta1_power_1save_15/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
¨
save_15/Assign_2Assignbeta2_powersave_15/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Š
save_15/Assign_3Assignbeta2_power_1save_15/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ż
save_15/Assign_4Assignpi/dense/biassave_15/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_15/Assign_5Assignpi/dense/bias/Adamsave_15/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_15/Assign_6Assignpi/dense/bias/Adam_1save_15/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_15/Assign_7Assignpi/dense/kernelsave_15/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_15/Assign_8Assignpi/dense/kernel/Adamsave_15/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ž
save_15/Assign_9Assignpi/dense/kernel/Adam_1save_15/RestoreV2:9*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ľ
save_15/Assign_10Assignpi/dense_1/biassave_15/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_15/Assign_11Assignpi/dense_1/bias/Adamsave_15/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_15/Assign_12Assignpi/dense_1/bias/Adam_1save_15/RestoreV2:12*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ž
save_15/Assign_13Assignpi/dense_1/kernelsave_15/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_15/Assign_14Assignpi/dense_1/kernel/Adamsave_15/RestoreV2:14*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Ĺ
save_15/Assign_15Assignpi/dense_1/kernel/Adam_1save_15/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_15/Assign_16Assignpi/dense_2/biassave_15/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_15/Assign_17Assignpi/dense_2/bias/Adamsave_15/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
ť
save_15/Assign_18Assignpi/dense_2/bias/Adam_1save_15/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
˝
save_15/Assign_19Assignpi/dense_2/kernelsave_15/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_15/Assign_20Assignpi/dense_2/kernel/Adamsave_15/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_15/Assign_21Assignpi/dense_2/kernel/Adam_1save_15/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ş
save_15/Assign_22Assign
pi/log_stdsave_15/RestoreV2:22*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_15/Assign_23Assignpi/log_std/Adamsave_15/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
ą
save_15/Assign_24Assignpi/log_std/Adam_1save_15/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_15/Assign_25Assignv/dense/biassave_15/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_15/Assign_26Assignv/dense/bias/Adamsave_15/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_15/Assign_27Assignv/dense/bias/Adam_1save_15/RestoreV2:27*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ˇ
save_15/Assign_28Assignv/dense/kernelsave_15/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_15/Assign_29Assignv/dense/kernel/Adamsave_15/RestoreV2:29*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ž
save_15/Assign_30Assignv/dense/kernel/Adam_1save_15/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_15/Assign_31Assignv/dense_1/biassave_15/RestoreV2:31*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_15/Assign_32Assignv/dense_1/bias/Adamsave_15/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_15/Assign_33Assignv/dense_1/bias/Adam_1save_15/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_15/Assign_34Assignv/dense_1/kernelsave_15/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_15/Assign_35Assignv/dense_1/kernel/Adamsave_15/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_15/Assign_36Assignv/dense_1/kernel/Adam_1save_15/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_15/Assign_37Assignv/dense_2/biassave_15/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_15/Assign_38Assignv/dense_2/bias/Adamsave_15/RestoreV2:38*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
š
save_15/Assign_39Assignv/dense_2/bias/Adam_1save_15/RestoreV2:39*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_15/Assign_40Assignv/dense_2/kernelsave_15/RestoreV2:40*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Ŕ
save_15/Assign_41Assignv/dense_2/kernel/Adamsave_15/RestoreV2:41*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Â
save_15/Assign_42Assignv/dense_2/kernel/Adam_1save_15/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
í
save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_38^save_15/Assign_39^save_15/Assign_4^save_15/Assign_40^save_15/Assign_41^save_15/Assign_42^save_15/Assign_5^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard
\
save_16/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_16/filenamePlaceholderWithDefaultsave_16/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_16/ConstPlaceholderWithDefaultsave_16/filename*
dtype0*
_output_shapes
: *
shape: 

save_16/StringJoin/inputs_1Const*<
value3B1 B+_temp_920a3d6aebd9477f87857b5f0653251b/part*
dtype0*
_output_shapes
: 
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_16/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_16/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
ť
save_16/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_16/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2*
T0**
_class 
loc:@save_16/ShardedFilename*
_output_shapes
: 
Ś
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(

save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
T0*
_output_shapes
: 
ž
save_16/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_16/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ń
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_16/AssignAssignbeta1_powersave_16/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Š
save_16/Assign_1Assignbeta1_power_1save_16/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
¨
save_16/Assign_2Assignbeta2_powersave_16/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Š
save_16/Assign_3Assignbeta2_power_1save_16/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ż
save_16/Assign_4Assignpi/dense/biassave_16/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_16/Assign_5Assignpi/dense/bias/Adamsave_16/RestoreV2:5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ś
save_16/Assign_6Assignpi/dense/bias/Adam_1save_16/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_16/Assign_7Assignpi/dense/kernelsave_16/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_16/Assign_8Assignpi/dense/kernel/Adamsave_16/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ž
save_16/Assign_9Assignpi/dense/kernel/Adam_1save_16/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ľ
save_16/Assign_10Assignpi/dense_1/biassave_16/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_16/Assign_11Assignpi/dense_1/bias/Adamsave_16/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_16/Assign_12Assignpi/dense_1/bias/Adam_1save_16/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ž
save_16/Assign_13Assignpi/dense_1/kernelsave_16/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_16/Assign_14Assignpi/dense_1/kernel/Adamsave_16/RestoreV2:14*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
Ĺ
save_16/Assign_15Assignpi/dense_1/kernel/Adam_1save_16/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_16/Assign_16Assignpi/dense_2/biassave_16/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_16/Assign_17Assignpi/dense_2/bias/Adamsave_16/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ť
save_16/Assign_18Assignpi/dense_2/bias/Adam_1save_16/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
˝
save_16/Assign_19Assignpi/dense_2/kernelsave_16/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_16/Assign_20Assignpi/dense_2/kernel/Adamsave_16/RestoreV2:20*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ä
save_16/Assign_21Assignpi/dense_2/kernel/Adam_1save_16/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ş
save_16/Assign_22Assign
pi/log_stdsave_16/RestoreV2:22*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_16/Assign_23Assignpi/log_std/Adamsave_16/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
ą
save_16/Assign_24Assignpi/log_std/Adam_1save_16/RestoreV2:24*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_16/Assign_25Assignv/dense/biassave_16/RestoreV2:25*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_16/Assign_26Assignv/dense/bias/Adamsave_16/RestoreV2:26*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_16/Assign_27Assignv/dense/bias/Adam_1save_16/RestoreV2:27*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
ˇ
save_16/Assign_28Assignv/dense/kernelsave_16/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_16/Assign_29Assignv/dense/kernel/Adamsave_16/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ž
save_16/Assign_30Assignv/dense/kernel/Adam_1save_16/RestoreV2:30*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ł
save_16/Assign_31Assignv/dense_1/biassave_16/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
¸
save_16/Assign_32Assignv/dense_1/bias/Adamsave_16/RestoreV2:32*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ş
save_16/Assign_33Assignv/dense_1/bias/Adam_1save_16/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_16/Assign_34Assignv/dense_1/kernelsave_16/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_16/Assign_35Assignv/dense_1/kernel/Adamsave_16/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_16/Assign_36Assignv/dense_1/kernel/Adam_1save_16/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_16/Assign_37Assignv/dense_2/biassave_16/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_16/Assign_38Assignv/dense_2/bias/Adamsave_16/RestoreV2:38*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_16/Assign_39Assignv/dense_2/bias/Adam_1save_16/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_16/Assign_40Assignv/dense_2/kernelsave_16/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_16/Assign_41Assignv/dense_2/kernel/Adamsave_16/RestoreV2:41*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_16/Assign_42Assignv/dense_2/kernel/Adam_1save_16/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
í
save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_20^save_16/Assign_21^save_16/Assign_22^save_16/Assign_23^save_16/Assign_24^save_16/Assign_25^save_16/Assign_26^save_16/Assign_27^save_16/Assign_28^save_16/Assign_29^save_16/Assign_3^save_16/Assign_30^save_16/Assign_31^save_16/Assign_32^save_16/Assign_33^save_16/Assign_34^save_16/Assign_35^save_16/Assign_36^save_16/Assign_37^save_16/Assign_38^save_16/Assign_39^save_16/Assign_4^save_16/Assign_40^save_16/Assign_41^save_16/Assign_42^save_16/Assign_5^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard
\
save_17/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_17/filenamePlaceholderWithDefaultsave_17/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_17/ConstPlaceholderWithDefaultsave_17/filename*
dtype0*
_output_shapes
: *
shape: 

save_17/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_bd9dd975e3c44dc88d6cfa4b747883da/part
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_17/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_17/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
ť
save_17/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_17/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_17/ShardedFilename
Ś
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilename^save_17/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(

save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency*
T0*
_output_shapes
: 
ž
save_17/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_17/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ń
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_17/AssignAssignbeta1_powersave_17/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Š
save_17/Assign_1Assignbeta1_power_1save_17/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
¨
save_17/Assign_2Assignbeta2_powersave_17/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Š
save_17/Assign_3Assignbeta2_power_1save_17/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ż
save_17/Assign_4Assignpi/dense/biassave_17/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
´
save_17/Assign_5Assignpi/dense/bias/Adamsave_17/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_17/Assign_6Assignpi/dense/bias/Adam_1save_17/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_17/Assign_7Assignpi/dense/kernelsave_17/RestoreV2:7*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ź
save_17/Assign_8Assignpi/dense/kernel/Adamsave_17/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ž
save_17/Assign_9Assignpi/dense/kernel/Adam_1save_17/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ľ
save_17/Assign_10Assignpi/dense_1/biassave_17/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ş
save_17/Assign_11Assignpi/dense_1/bias/Adamsave_17/RestoreV2:11*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ź
save_17/Assign_12Assignpi/dense_1/bias/Adam_1save_17/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ž
save_17/Assign_13Assignpi/dense_1/kernelsave_17/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_17/Assign_14Assignpi/dense_1/kernel/Adamsave_17/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ĺ
save_17/Assign_15Assignpi/dense_1/kernel/Adam_1save_17/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

´
save_17/Assign_16Assignpi/dense_2/biassave_17/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_17/Assign_17Assignpi/dense_2/bias/Adamsave_17/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_17/Assign_18Assignpi/dense_2/bias/Adam_1save_17/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
˝
save_17/Assign_19Assignpi/dense_2/kernelsave_17/RestoreV2:19*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Â
save_17/Assign_20Assignpi/dense_2/kernel/Adamsave_17/RestoreV2:20*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ä
save_17/Assign_21Assignpi/dense_2/kernel/Adam_1save_17/RestoreV2:21*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Ş
save_17/Assign_22Assign
pi/log_stdsave_17/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
Ż
save_17/Assign_23Assignpi/log_std/Adamsave_17/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
ą
save_17/Assign_24Assignpi/log_std/Adam_1save_17/RestoreV2:24*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_17/Assign_25Assignv/dense/biassave_17/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_17/Assign_26Assignv/dense/bias/Adamsave_17/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_17/Assign_27Assignv/dense/bias/Adam_1save_17/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_17/Assign_28Assignv/dense/kernelsave_17/RestoreV2:28*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_17/Assign_29Assignv/dense/kernel/Adamsave_17/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_17/Assign_30Assignv/dense/kernel/Adam_1save_17/RestoreV2:30*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
ł
save_17/Assign_31Assignv/dense_1/biassave_17/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_17/Assign_32Assignv/dense_1/bias/Adamsave_17/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_17/Assign_33Assignv/dense_1/bias/Adam_1save_17/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ź
save_17/Assign_34Assignv/dense_1/kernelsave_17/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_17/Assign_35Assignv/dense_1/kernel/Adamsave_17/RestoreV2:35*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_17/Assign_36Assignv/dense_1/kernel/Adam_1save_17/RestoreV2:36*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˛
save_17/Assign_37Assignv/dense_2/biassave_17/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_17/Assign_38Assignv/dense_2/bias/Adamsave_17/RestoreV2:38*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_17/Assign_39Assignv/dense_2/bias/Adam_1save_17/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_17/Assign_40Assignv/dense_2/kernelsave_17/RestoreV2:40*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Ŕ
save_17/Assign_41Assignv/dense_2/kernel/Adamsave_17/RestoreV2:41*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Â
save_17/Assign_42Assignv/dense_2/kernel/Adam_1save_17/RestoreV2:42*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
í
save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_16^save_17/Assign_17^save_17/Assign_18^save_17/Assign_19^save_17/Assign_2^save_17/Assign_20^save_17/Assign_21^save_17/Assign_22^save_17/Assign_23^save_17/Assign_24^save_17/Assign_25^save_17/Assign_26^save_17/Assign_27^save_17/Assign_28^save_17/Assign_29^save_17/Assign_3^save_17/Assign_30^save_17/Assign_31^save_17/Assign_32^save_17/Assign_33^save_17/Assign_34^save_17/Assign_35^save_17/Assign_36^save_17/Assign_37^save_17/Assign_38^save_17/Assign_39^save_17/Assign_4^save_17/Assign_40^save_17/Assign_41^save_17/Assign_42^save_17/Assign_5^save_17/Assign_6^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9
3
save_17/restore_allNoOp^save_17/restore_shard
\
save_18/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_18/filenamePlaceholderWithDefaultsave_18/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_18/ConstPlaceholderWithDefaultsave_18/filename*
dtype0*
_output_shapes
: *
shape: 

save_18/StringJoin/inputs_1Const*<
value3B1 B+_temp_5300a6d4e051496d86db7ec79f89f2ac/part*
dtype0*
_output_shapes
: 
~
save_18/StringJoin
StringJoinsave_18/Constsave_18/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_18/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_18/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_18/ShardedFilenameShardedFilenamesave_18/StringJoinsave_18/ShardedFilename/shardsave_18/num_shards*
_output_shapes
: 
ť
save_18/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_18/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_18/SaveV2SaveV2save_18/ShardedFilenamesave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_18/control_dependencyIdentitysave_18/ShardedFilename^save_18/SaveV2*
T0**
_class 
loc:@save_18/ShardedFilename*
_output_shapes
: 
Ś
.save_18/MergeV2Checkpoints/checkpoint_prefixesPacksave_18/ShardedFilename^save_18/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_18/MergeV2CheckpointsMergeV2Checkpoints.save_18/MergeV2Checkpoints/checkpoint_prefixessave_18/Const*
delete_old_dirs(

save_18/IdentityIdentitysave_18/Const^save_18/MergeV2Checkpoints^save_18/control_dependency*
T0*
_output_shapes
: 
ž
save_18/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_18/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ń
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_18/AssignAssignbeta1_powersave_18/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Š
save_18/Assign_1Assignbeta1_power_1save_18/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
¨
save_18/Assign_2Assignbeta2_powersave_18/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Š
save_18/Assign_3Assignbeta2_power_1save_18/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ż
save_18/Assign_4Assignpi/dense/biassave_18/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
´
save_18/Assign_5Assignpi/dense/bias/Adamsave_18/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_18/Assign_6Assignpi/dense/bias/Adam_1save_18/RestoreV2:6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ˇ
save_18/Assign_7Assignpi/dense/kernelsave_18/RestoreV2:7*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ź
save_18/Assign_8Assignpi/dense/kernel/Adamsave_18/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_18/Assign_9Assignpi/dense/kernel/Adam_1save_18/RestoreV2:9*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ľ
save_18/Assign_10Assignpi/dense_1/biassave_18/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ş
save_18/Assign_11Assignpi/dense_1/bias/Adamsave_18/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_18/Assign_12Assignpi/dense_1/bias/Adam_1save_18/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ž
save_18/Assign_13Assignpi/dense_1/kernelsave_18/RestoreV2:13*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_18/Assign_14Assignpi/dense_1/kernel/Adamsave_18/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ĺ
save_18/Assign_15Assignpi/dense_1/kernel/Adam_1save_18/RestoreV2:15*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
´
save_18/Assign_16Assignpi/dense_2/biassave_18/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
š
save_18/Assign_17Assignpi/dense_2/bias/Adamsave_18/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_18/Assign_18Assignpi/dense_2/bias/Adam_1save_18/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
˝
save_18/Assign_19Assignpi/dense_2/kernelsave_18/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_18/Assign_20Assignpi/dense_2/kernel/Adamsave_18/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_18/Assign_21Assignpi/dense_2/kernel/Adam_1save_18/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ş
save_18/Assign_22Assign
pi/log_stdsave_18/RestoreV2:22*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_18/Assign_23Assignpi/log_std/Adamsave_18/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
ą
save_18/Assign_24Assignpi/log_std/Adam_1save_18/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_18/Assign_25Assignv/dense/biassave_18/RestoreV2:25*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@v/dense/bias
´
save_18/Assign_26Assignv/dense/bias/Adamsave_18/RestoreV2:26*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_18/Assign_27Assignv/dense/bias/Adam_1save_18/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_18/Assign_28Assignv/dense/kernelsave_18/RestoreV2:28*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_18/Assign_29Assignv/dense/kernel/Adamsave_18/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_18/Assign_30Assignv/dense/kernel/Adam_1save_18/RestoreV2:30*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ł
save_18/Assign_31Assignv/dense_1/biassave_18/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
¸
save_18/Assign_32Assignv/dense_1/bias/Adamsave_18/RestoreV2:32*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save_18/Assign_33Assignv/dense_1/bias/Adam_1save_18/RestoreV2:33*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_18/Assign_34Assignv/dense_1/kernelsave_18/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_18/Assign_35Assignv/dense_1/kernel/Adamsave_18/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_18/Assign_36Assignv/dense_1/kernel/Adam_1save_18/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_18/Assign_37Assignv/dense_2/biassave_18/RestoreV2:37*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ˇ
save_18/Assign_38Assignv/dense_2/bias/Adamsave_18/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_18/Assign_39Assignv/dense_2/bias/Adam_1save_18/RestoreV2:39*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
ť
save_18/Assign_40Assignv/dense_2/kernelsave_18/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_18/Assign_41Assignv/dense_2/kernel/Adamsave_18/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_18/Assign_42Assignv/dense_2/kernel/Adam_1save_18/RestoreV2:42*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
í
save_18/restore_shardNoOp^save_18/Assign^save_18/Assign_1^save_18/Assign_10^save_18/Assign_11^save_18/Assign_12^save_18/Assign_13^save_18/Assign_14^save_18/Assign_15^save_18/Assign_16^save_18/Assign_17^save_18/Assign_18^save_18/Assign_19^save_18/Assign_2^save_18/Assign_20^save_18/Assign_21^save_18/Assign_22^save_18/Assign_23^save_18/Assign_24^save_18/Assign_25^save_18/Assign_26^save_18/Assign_27^save_18/Assign_28^save_18/Assign_29^save_18/Assign_3^save_18/Assign_30^save_18/Assign_31^save_18/Assign_32^save_18/Assign_33^save_18/Assign_34^save_18/Assign_35^save_18/Assign_36^save_18/Assign_37^save_18/Assign_38^save_18/Assign_39^save_18/Assign_4^save_18/Assign_40^save_18/Assign_41^save_18/Assign_42^save_18/Assign_5^save_18/Assign_6^save_18/Assign_7^save_18/Assign_8^save_18/Assign_9
3
save_18/restore_allNoOp^save_18/restore_shard
\
save_19/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_19/filenamePlaceholderWithDefaultsave_19/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_19/ConstPlaceholderWithDefaultsave_19/filename*
dtype0*
_output_shapes
: *
shape: 

save_19/StringJoin/inputs_1Const*<
value3B1 B+_temp_176a58b7cd2c4ab3a101c160c411fe06/part*
dtype0*
_output_shapes
: 
~
save_19/StringJoin
StringJoinsave_19/Constsave_19/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_19/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_19/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_19/ShardedFilenameShardedFilenamesave_19/StringJoinsave_19/ShardedFilename/shardsave_19/num_shards*
_output_shapes
: 
ť
save_19/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_19/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_19/SaveV2SaveV2save_19/ShardedFilenamesave_19/SaveV2/tensor_namessave_19/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_19/control_dependencyIdentitysave_19/ShardedFilename^save_19/SaveV2*
T0**
_class 
loc:@save_19/ShardedFilename*
_output_shapes
: 
Ś
.save_19/MergeV2Checkpoints/checkpoint_prefixesPacksave_19/ShardedFilename^save_19/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_19/MergeV2CheckpointsMergeV2Checkpoints.save_19/MergeV2Checkpoints/checkpoint_prefixessave_19/Const*
delete_old_dirs(

save_19/IdentityIdentitysave_19/Const^save_19/MergeV2Checkpoints^save_19/control_dependency*
_output_shapes
: *
T0
ž
save_19/RestoreV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ż
"save_19/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ń
save_19/RestoreV2	RestoreV2save_19/Constsave_19/RestoreV2/tensor_names"save_19/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_19/AssignAssignbeta1_powersave_19/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Š
save_19/Assign_1Assignbeta1_power_1save_19/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
¨
save_19/Assign_2Assignbeta2_powersave_19/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Š
save_19/Assign_3Assignbeta2_power_1save_19/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
Ż
save_19/Assign_4Assignpi/dense/biassave_19/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
´
save_19/Assign_5Assignpi/dense/bias/Adamsave_19/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ś
save_19/Assign_6Assignpi/dense/bias/Adam_1save_19/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_19/Assign_7Assignpi/dense/kernelsave_19/RestoreV2:7*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ź
save_19/Assign_8Assignpi/dense/kernel/Adamsave_19/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ž
save_19/Assign_9Assignpi/dense/kernel/Adam_1save_19/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ľ
save_19/Assign_10Assignpi/dense_1/biassave_19/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_19/Assign_11Assignpi/dense_1/bias/Adamsave_19/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ź
save_19/Assign_12Assignpi/dense_1/bias/Adam_1save_19/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ž
save_19/Assign_13Assignpi/dense_1/kernelsave_19/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ă
save_19/Assign_14Assignpi/dense_1/kernel/Adamsave_19/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ĺ
save_19/Assign_15Assignpi/dense_1/kernel/Adam_1save_19/RestoreV2:15*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
´
save_19/Assign_16Assignpi/dense_2/biassave_19/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_19/Assign_17Assignpi/dense_2/bias/Adamsave_19/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_19/Assign_18Assignpi/dense_2/bias/Adam_1save_19/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
˝
save_19/Assign_19Assignpi/dense_2/kernelsave_19/RestoreV2:19*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
Â
save_19/Assign_20Assignpi/dense_2/kernel/Adamsave_19/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_19/Assign_21Assignpi/dense_2/kernel/Adam_1save_19/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ş
save_19/Assign_22Assign
pi/log_stdsave_19/RestoreV2:22*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_19/Assign_23Assignpi/log_std/Adamsave_19/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
ą
save_19/Assign_24Assignpi/log_std/Adam_1save_19/RestoreV2:24*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
Ż
save_19/Assign_25Assignv/dense/biassave_19/RestoreV2:25*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
´
save_19/Assign_26Assignv/dense/bias/Adamsave_19/RestoreV2:26*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_19/Assign_27Assignv/dense/bias/Adam_1save_19/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_19/Assign_28Assignv/dense/kernelsave_19/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_19/Assign_29Assignv/dense/kernel/Adamsave_19/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ž
save_19/Assign_30Assignv/dense/kernel/Adam_1save_19/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ł
save_19/Assign_31Assignv/dense_1/biassave_19/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:
¸
save_19/Assign_32Assignv/dense_1/bias/Adamsave_19/RestoreV2:32*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ş
save_19/Assign_33Assignv/dense_1/bias/Adam_1save_19/RestoreV2:33*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_19/Assign_34Assignv/dense_1/kernelsave_19/RestoreV2:34*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
Á
save_19/Assign_35Assignv/dense_1/kernel/Adamsave_19/RestoreV2:35*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_19/Assign_36Assignv/dense_1/kernel/Adam_1save_19/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_19/Assign_37Assignv/dense_2/biassave_19/RestoreV2:37*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ˇ
save_19/Assign_38Assignv/dense_2/bias/Adamsave_19/RestoreV2:38*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_19/Assign_39Assignv/dense_2/bias/Adam_1save_19/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_19/Assign_40Assignv/dense_2/kernelsave_19/RestoreV2:40*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Ŕ
save_19/Assign_41Assignv/dense_2/kernel/Adamsave_19/RestoreV2:41*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_19/Assign_42Assignv/dense_2/kernel/Adam_1save_19/RestoreV2:42*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
í
save_19/restore_shardNoOp^save_19/Assign^save_19/Assign_1^save_19/Assign_10^save_19/Assign_11^save_19/Assign_12^save_19/Assign_13^save_19/Assign_14^save_19/Assign_15^save_19/Assign_16^save_19/Assign_17^save_19/Assign_18^save_19/Assign_19^save_19/Assign_2^save_19/Assign_20^save_19/Assign_21^save_19/Assign_22^save_19/Assign_23^save_19/Assign_24^save_19/Assign_25^save_19/Assign_26^save_19/Assign_27^save_19/Assign_28^save_19/Assign_29^save_19/Assign_3^save_19/Assign_30^save_19/Assign_31^save_19/Assign_32^save_19/Assign_33^save_19/Assign_34^save_19/Assign_35^save_19/Assign_36^save_19/Assign_37^save_19/Assign_38^save_19/Assign_39^save_19/Assign_4^save_19/Assign_40^save_19/Assign_41^save_19/Assign_42^save_19/Assign_5^save_19/Assign_6^save_19/Assign_7^save_19/Assign_8^save_19/Assign_9
3
save_19/restore_allNoOp^save_19/restore_shard
\
save_20/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_20/filenamePlaceholderWithDefaultsave_20/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_20/ConstPlaceholderWithDefaultsave_20/filename*
dtype0*
_output_shapes
: *
shape: 

save_20/StringJoin/inputs_1Const*<
value3B1 B+_temp_3f9a4e1f7920483ea6cbea54ed27876f/part*
dtype0*
_output_shapes
: 
~
save_20/StringJoin
StringJoinsave_20/Constsave_20/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_20/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_20/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_20/ShardedFilenameShardedFilenamesave_20/StringJoinsave_20/ShardedFilename/shardsave_20/num_shards*
_output_shapes
: 
ť
save_20/SaveV2/tensor_namesConst*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:+
ź
save_20/SaveV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+

save_20/SaveV2SaveV2save_20/ShardedFilenamesave_20/SaveV2/tensor_namessave_20/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*9
dtypes/
-2+

save_20/control_dependencyIdentitysave_20/ShardedFilename^save_20/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_20/ShardedFilename
Ś
.save_20/MergeV2Checkpoints/checkpoint_prefixesPacksave_20/ShardedFilename^save_20/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_20/MergeV2CheckpointsMergeV2Checkpoints.save_20/MergeV2Checkpoints/checkpoint_prefixessave_20/Const*
delete_old_dirs(

save_20/IdentityIdentitysave_20/Const^save_20/MergeV2Checkpoints^save_20/control_dependency*
T0*
_output_shapes
: 
ž
save_20/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:+*ë
valueáBŢ+Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
ż
"save_20/RestoreV2/shape_and_slicesConst*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+
ń
save_20/RestoreV2	RestoreV2save_20/Constsave_20/RestoreV2/tensor_names"save_20/RestoreV2/shape_and_slices*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+
¤
save_20/AssignAssignbeta1_powersave_20/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Š
save_20/Assign_1Assignbeta1_power_1save_20/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
¨
save_20/Assign_2Assignbeta2_powersave_20/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Š
save_20/Assign_3Assignbeta2_power_1save_20/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ż
save_20/Assign_4Assignpi/dense/biassave_20/RestoreV2:4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
´
save_20/Assign_5Assignpi/dense/bias/Adamsave_20/RestoreV2:5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ś
save_20/Assign_6Assignpi/dense/bias/Adam_1save_20/RestoreV2:6*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
ˇ
save_20/Assign_7Assignpi/dense/kernelsave_20/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_20/Assign_8Assignpi/dense/kernel/Adamsave_20/RestoreV2:8*
validate_shape(*
_output_shapes
:	'*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
ž
save_20/Assign_9Assignpi/dense/kernel/Adam_1save_20/RestoreV2:9*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ľ
save_20/Assign_10Assignpi/dense_1/biassave_20/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ş
save_20/Assign_11Assignpi/dense_1/bias/Adamsave_20/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ź
save_20/Assign_12Assignpi/dense_1/bias/Adam_1save_20/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes	
:
ž
save_20/Assign_13Assignpi/dense_1/kernelsave_20/RestoreV2:13*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ă
save_20/Assign_14Assignpi/dense_1/kernel/Adamsave_20/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ĺ
save_20/Assign_15Assignpi/dense_1/kernel/Adam_1save_20/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
´
save_20/Assign_16Assignpi/dense_2/biassave_20/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
š
save_20/Assign_17Assignpi/dense_2/bias/Adamsave_20/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_20/Assign_18Assignpi/dense_2/bias/Adam_1save_20/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
˝
save_20/Assign_19Assignpi/dense_2/kernelsave_20/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Â
save_20/Assign_20Assignpi/dense_2/kernel/Adamsave_20/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ä
save_20/Assign_21Assignpi/dense_2/kernel/Adam_1save_20/RestoreV2:21*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
Ş
save_20/Assign_22Assign
pi/log_stdsave_20/RestoreV2:22*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_20/Assign_23Assignpi/log_std/Adamsave_20/RestoreV2:23*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
ą
save_20/Assign_24Assignpi/log_std/Adam_1save_20/RestoreV2:24*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:
Ż
save_20/Assign_25Assignv/dense/biassave_20/RestoreV2:25*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
´
save_20/Assign_26Assignv/dense/bias/Adamsave_20/RestoreV2:26*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ś
save_20/Assign_27Assignv/dense/bias/Adam_1save_20/RestoreV2:27*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes	
:
ˇ
save_20/Assign_28Assignv/dense/kernelsave_20/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'
ź
save_20/Assign_29Assignv/dense/kernel/Adamsave_20/RestoreV2:29*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ž
save_20/Assign_30Assignv/dense/kernel/Adam_1save_20/RestoreV2:30*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	'*
use_locking(
ł
save_20/Assign_31Assignv/dense_1/biassave_20/RestoreV2:31*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
¸
save_20/Assign_32Assignv/dense_1/bias/Adamsave_20/RestoreV2:32*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ş
save_20/Assign_33Assignv/dense_1/bias/Adam_1save_20/RestoreV2:33*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
ź
save_20/Assign_34Assignv/dense_1/kernelsave_20/RestoreV2:34*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Á
save_20/Assign_35Assignv/dense_1/kernel/Adamsave_20/RestoreV2:35*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
Ă
save_20/Assign_36Assignv/dense_1/kernel/Adam_1save_20/RestoreV2:36*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(* 
_output_shapes
:

˛
save_20/Assign_37Assignv/dense_2/biassave_20/RestoreV2:37*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ˇ
save_20/Assign_38Assignv/dense_2/bias/Adamsave_20/RestoreV2:38*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
š
save_20/Assign_39Assignv/dense_2/bias/Adam_1save_20/RestoreV2:39*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
ť
save_20/Assign_40Assignv/dense_2/kernelsave_20/RestoreV2:40*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Ŕ
save_20/Assign_41Assignv/dense_2/kernel/Adamsave_20/RestoreV2:41*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
Â
save_20/Assign_42Assignv/dense_2/kernel/Adam_1save_20/RestoreV2:42*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes
:	
í
save_20/restore_shardNoOp^save_20/Assign^save_20/Assign_1^save_20/Assign_10^save_20/Assign_11^save_20/Assign_12^save_20/Assign_13^save_20/Assign_14^save_20/Assign_15^save_20/Assign_16^save_20/Assign_17^save_20/Assign_18^save_20/Assign_19^save_20/Assign_2^save_20/Assign_20^save_20/Assign_21^save_20/Assign_22^save_20/Assign_23^save_20/Assign_24^save_20/Assign_25^save_20/Assign_26^save_20/Assign_27^save_20/Assign_28^save_20/Assign_29^save_20/Assign_3^save_20/Assign_30^save_20/Assign_31^save_20/Assign_32^save_20/Assign_33^save_20/Assign_34^save_20/Assign_35^save_20/Assign_36^save_20/Assign_37^save_20/Assign_38^save_20/Assign_39^save_20/Assign_4^save_20/Assign_40^save_20/Assign_41^save_20/Assign_42^save_20/Assign_5^save_20/Assign_6^save_20/Assign_7^save_20/Assign_8^save_20/Assign_9
3
save_20/restore_allNoOp^save_20/restore_shard "E
save_20/Const:0save_20/Identity:0save_20/restore_all (5 @F8"
train_op

Adam
Adam_1"(
	variables((
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
o
v/dense/kernel:0v/dense/kernel/Assignv/dense/kernel/read:02+v/dense/kernel/Initializer/random_uniform:08
^
v/dense/bias:0v/dense/bias/Assignv/dense/bias/read:02 v/dense/bias/Initializer/zeros:08
w
v/dense_1/kernel:0v/dense_1/kernel/Assignv/dense_1/kernel/read:02-v/dense_1/kernel/Initializer/random_uniform:08
f
v/dense_1/bias:0v/dense_1/bias/Assignv/dense_1/bias/read:02"v/dense_1/bias/Initializer/zeros:08
w
v/dense_2/kernel:0v/dense_2/kernel/Assignv/dense_2/kernel/read:02-v/dense_2/kernel/Initializer/random_uniform:08
f
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
|
pi/dense/kernel/Adam:0pi/dense/kernel/Adam/Assignpi/dense/kernel/Adam/read:02(pi/dense/kernel/Adam/Initializer/zeros:0

pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0

pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0

pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0

pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0

pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0

pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0

pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
h
pi/log_std/Adam:0pi/log_std/Adam/Assignpi/log_std/Adam/read:02#pi/log_std/Adam/Initializer/zeros:0
p
pi/log_std/Adam_1:0pi/log_std/Adam_1/Assignpi/log_std/Adam_1/read:02%pi/log_std/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
x
v/dense/kernel/Adam:0v/dense/kernel/Adam/Assignv/dense/kernel/Adam/read:02'v/dense/kernel/Adam/Initializer/zeros:0

v/dense/kernel/Adam_1:0v/dense/kernel/Adam_1/Assignv/dense/kernel/Adam_1/read:02)v/dense/kernel/Adam_1/Initializer/zeros:0
p
v/dense/bias/Adam:0v/dense/bias/Adam/Assignv/dense/bias/Adam/read:02%v/dense/bias/Adam/Initializer/zeros:0
x
v/dense/bias/Adam_1:0v/dense/bias/Adam_1/Assignv/dense/bias/Adam_1/read:02'v/dense/bias/Adam_1/Initializer/zeros:0

v/dense_1/kernel/Adam:0v/dense_1/kernel/Adam/Assignv/dense_1/kernel/Adam/read:02)v/dense_1/kernel/Adam/Initializer/zeros:0

v/dense_1/kernel/Adam_1:0v/dense_1/kernel/Adam_1/Assignv/dense_1/kernel/Adam_1/read:02+v/dense_1/kernel/Adam_1/Initializer/zeros:0
x
v/dense_1/bias/Adam:0v/dense_1/bias/Adam/Assignv/dense_1/bias/Adam/read:02'v/dense_1/bias/Adam/Initializer/zeros:0

v/dense_1/bias/Adam_1:0v/dense_1/bias/Adam_1/Assignv/dense_1/bias/Adam_1/read:02)v/dense_1/bias/Adam_1/Initializer/zeros:0

v/dense_2/kernel/Adam:0v/dense_2/kernel/Adam/Assignv/dense_2/kernel/Adam/read:02)v/dense_2/kernel/Adam/Initializer/zeros:0

v/dense_2/kernel/Adam_1:0v/dense_2/kernel/Adam_1/Assignv/dense_2/kernel/Adam_1/read:02+v/dense_2/kernel/Adam_1/Initializer/zeros:0
x
v/dense_2/bias/Adam:0v/dense_2/bias/Adam/Assignv/dense_2/bias/Adam/read:02'v/dense_2/bias/Adam/Initializer/zeros:0

v/dense_2/bias/Adam_1:0v/dense_2/bias/Adam_1/Assignv/dense_2/bias/Adam_1/read:02)v/dense_2/bias/Adam_1/Initializer/zeros:0"­
trainable_variables
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
o
v/dense/kernel:0v/dense/kernel/Assignv/dense/kernel/read:02+v/dense/kernel/Initializer/random_uniform:08
^
v/dense/bias:0v/dense/bias/Assignv/dense/bias/read:02 v/dense/bias/Initializer/zeros:08
w
v/dense_1/kernel:0v/dense_1/kernel/Assignv/dense_1/kernel/read:02-v/dense_1/kernel/Initializer/random_uniform:08
f
v/dense_1/bias:0v/dense_1/bias/Assignv/dense_1/bias/read:02"v/dense_1/bias/Initializer/zeros:08
w
v/dense_2/kernel:0v/dense_2/kernel/Assignv/dense_2/kernel/read:02-v/dense_2/kernel/Initializer/random_uniform:08
f
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08*§
serving_default
)
x$
Placeholder:0˙˙˙˙˙˙˙˙˙'#
v
v/Squeeze:0˙˙˙˙˙˙˙˙˙%
pi
pi/add:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict