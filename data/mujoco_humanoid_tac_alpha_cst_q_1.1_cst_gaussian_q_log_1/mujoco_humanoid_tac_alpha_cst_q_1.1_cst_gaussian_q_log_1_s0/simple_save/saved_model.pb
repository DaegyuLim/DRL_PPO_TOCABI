сј
Я)┤)
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
2	ђљ
Ь
	ApplyAdam
var"Tђ	
m"Tђ	
v"Tђ
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"Tђ" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"Tђ

value"T

output_ref"Tђ"	
Ttype"
validate_shapebool("
use_lockingbool(ў
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
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
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
B
Equal
x"T
y"T
z
"
Ttype:
2	
љ
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

2	љ
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
;
Minimum
x"T
y"T
z"T"
Ttype:

2	љ
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
Ї
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ё
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
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
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
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
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
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
ref"dtypeђ"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ѕ
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12b'v1.13.1-0-g6612da8951'от
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:         I*
shape:         I
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
p
Placeholder_2Placeholder*
shape:         I*
dtype0*'
_output_shapes
:         I
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
N
Placeholder_5Placeholder*
shape: *
dtype0*
_output_shapes
: 
N
Placeholder_6Placeholder*
shape: *
dtype0*
_output_shapes
: 
R
main/pi/sub/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
main/pi/subSubmain/pi/sub/xPlaceholder_6*
T0*
_output_shapes
: 
»
5main/pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"I      *'
_class
loc:@main/pi/dense/kernel*
dtype0*
_output_shapes
:
А
3main/pi/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *%vЌй*'
_class
loc:@main/pi/dense/kernel*
dtype0*
_output_shapes
: 
А
3main/pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *%vЌ=*'
_class
loc:@main/pi/dense/kernel*
dtype0*
_output_shapes
: 
■
=main/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5main/pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	Iђ*

seed *
T0*'
_class
loc:@main/pi/dense/kernel*
seed2
Ь
3main/pi/dense/kernel/Initializer/random_uniform/subSub3main/pi/dense/kernel/Initializer/random_uniform/max3main/pi/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
: 
Ђ
3main/pi/dense/kernel/Initializer/random_uniform/mulMul=main/pi/dense/kernel/Initializer/random_uniform/RandomUniform3main/pi/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	Iђ
з
/main/pi/dense/kernel/Initializer/random_uniformAdd3main/pi/dense/kernel/Initializer/random_uniform/mul3main/pi/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	Iђ
│
main/pi/dense/kernel
VariableV2*
shared_name *'
_class
loc:@main/pi/dense/kernel*
	container *
shape:	Iђ*
dtype0*
_output_shapes
:	Iђ
У
main/pi/dense/kernel/AssignAssignmain/pi/dense/kernel/main/pi/dense/kernel/Initializer/random_uniform*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ*
use_locking(
ј
main/pi/dense/kernel/readIdentitymain/pi/dense/kernel*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	Iђ
д
4main/pi/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*%
_class
loc:@main/pi/dense/bias*
dtype0*
_output_shapes
:
ќ
*main/pi/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@main/pi/dense/bias*
dtype0*
_output_shapes
: 
ь
$main/pi/dense/bias/Initializer/zerosFill4main/pi/dense/bias/Initializer/zeros/shape_as_tensor*main/pi/dense/bias/Initializer/zeros/Const*
_output_shapes	
:ђ*
T0*

index_type0*%
_class
loc:@main/pi/dense/bias
Д
main/pi/dense/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape:ђ
М
main/pi/dense/bias/AssignAssignmain/pi/dense/bias$main/pi/dense/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ
ё
main/pi/dense/bias/readIdentitymain/pi/dense/bias*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:ђ
Ъ
main/pi/dense/MatMulMatMulPlaceholdermain/pi/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ў
main/pi/dense/BiasAddBiasAddmain/pi/dense/MatMulmain/pi/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
d
main/pi/dense/ReluRelumain/pi/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
│
7main/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
:
Ц
5main/pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *  ђй*)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
Ц
5main/pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  ђ=*)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
Ё
?main/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђђ*

seed *
T0*)
_class
loc:@main/pi/dense_1/kernel*
seed2
Ш
5main/pi/dense_1/kernel/Initializer/random_uniform/subSub5main/pi/dense_1/kernel/Initializer/random_uniform/max5main/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
: 
і
5main/pi/dense_1/kernel/Initializer/random_uniform/mulMul?main/pi/dense_1/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
ђђ*
T0*)
_class
loc:@main/pi/dense_1/kernel
Ч
1main/pi/dense_1/kernel/Initializer/random_uniformAdd5main/pi/dense_1/kernel/Initializer/random_uniform/mul5main/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
ђђ
╣
main/pi/dense_1/kernel
VariableV2*
shared_name *)
_class
loc:@main/pi/dense_1/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ
ы
main/pi/dense_1/kernel/AssignAssignmain/pi/dense_1/kernel1main/pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Ћ
main/pi/dense_1/kernel/readIdentitymain/pi/dense_1/kernel* 
_output_shapes
:
ђђ*
T0*)
_class
loc:@main/pi/dense_1/kernel
ъ
&main/pi/dense_1/bias/Initializer/zerosConst*
valueBђ*    *'
_class
loc:@main/pi/dense_1/bias*
dtype0*
_output_shapes	
:ђ
Ф
main/pi/dense_1/bias
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *'
_class
loc:@main/pi/dense_1/bias
█
main/pi/dense_1/bias/AssignAssignmain/pi/dense_1/bias&main/pi/dense_1/bias/Initializer/zeros*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
і
main/pi/dense_1/bias/readIdentitymain/pi/dense_1/bias*
_output_shapes	
:ђ*
T0*'
_class
loc:@main/pi/dense_1/bias
ф
main/pi/dense_1/MatMulMatMulmain/pi/dense/Relumain/pi/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
Ъ
main/pi/dense_1/BiasAddBiasAddmain/pi/dense_1/MatMulmain/pi/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
h
main/pi/dense_1/ReluRelumain/pi/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
│
7main/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *)
_class
loc:@main/pi/dense_2/kernel
Ц
5main/pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *ЬGОй*)
_class
loc:@main/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
Ц
5main/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ЬGО=*)
_class
loc:@main/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
ё
?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0*)
_class
loc:@main/pi/dense_2/kernel*
seed20
Ш
5main/pi/dense_2/kernel/Initializer/random_uniform/subSub5main/pi/dense_2/kernel/Initializer/random_uniform/max5main/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
: 
Ѕ
5main/pi/dense_2/kernel/Initializer/random_uniform/mulMul?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	ђ
ч
1main/pi/dense_2/kernel/Initializer/random_uniformAdd5main/pi/dense_2/kernel/Initializer/random_uniform/mul5main/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	ђ
и
main/pi/dense_2/kernel
VariableV2*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
­
main/pi/dense_2/kernel/AssignAssignmain/pi/dense_2/kernel1main/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
ћ
main/pi/dense_2/kernel/readIdentitymain/pi/dense_2/kernel*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/pi/dense_2/kernel
ю
&main/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/pi/dense_2/bias*
dtype0*
_output_shapes
:
Е
main/pi/dense_2/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container 
┌
main/pi/dense_2/bias/AssignAssignmain/pi/dense_2/bias&main/pi/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias
Ѕ
main/pi/dense_2/bias/readIdentitymain/pi/dense_2/bias*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:
Ф
main/pi/dense_2/MatMulMatMulmain/pi/dense_1/Relumain/pi/dense_2/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
ъ
main/pi/dense_2/BiasAddBiasAddmain/pi/dense_2/MatMulmain/pi/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
│
7main/pi/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *)
_class
loc:@main/pi/dense_3/kernel
Ц
5main/pi/dense_3/kernel/Initializer/random_uniform/minConst*
valueB
 *ЬGОй*)
_class
loc:@main/pi/dense_3/kernel*
dtype0*
_output_shapes
: 
Ц
5main/pi/dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *ЬGО=*)
_class
loc:@main/pi/dense_3/kernel*
dtype0*
_output_shapes
: 
ё
?main/pi/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0*)
_class
loc:@main/pi/dense_3/kernel*
seed2@
Ш
5main/pi/dense_3/kernel/Initializer/random_uniform/subSub5main/pi/dense_3/kernel/Initializer/random_uniform/max5main/pi/dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@main/pi/dense_3/kernel
Ѕ
5main/pi/dense_3/kernel/Initializer/random_uniform/mulMul?main/pi/dense_3/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/pi/dense_3/kernel
ч
1main/pi/dense_3/kernel/Initializer/random_uniformAdd5main/pi/dense_3/kernel/Initializer/random_uniform/mul5main/pi/dense_3/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_3/kernel*
_output_shapes
:	ђ
и
main/pi/dense_3/kernel
VariableV2*)
_class
loc:@main/pi/dense_3/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name 
­
main/pi/dense_3/kernel/AssignAssignmain/pi/dense_3/kernel1main/pi/dense_3/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@main/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ
ћ
main/pi/dense_3/kernel/readIdentitymain/pi/dense_3/kernel*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/pi/dense_3/kernel
ю
&main/pi/dense_3/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/pi/dense_3/bias*
dtype0*
_output_shapes
:
Е
main/pi/dense_3/bias
VariableV2*
shared_name *'
_class
loc:@main/pi/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes
:
┌
main/pi/dense_3/bias/AssignAssignmain/pi/dense_3/bias&main/pi/dense_3/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_3/bias
Ѕ
main/pi/dense_3/bias/readIdentitymain/pi/dense_3/bias*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_3/bias
Ф
main/pi/dense_3/MatMulMatMulmain/pi/dense_1/Relumain/pi/dense_3/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
ъ
main/pi/dense_3/BiasAddBiasAddmain/pi/dense_3/MatMulmain/pi/dense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
g
main/pi/dense_3/TanhTanhmain/pi/dense_3/BiasAdd*
T0*'
_output_shapes
:         
R
main/pi/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
i
main/pi/addAddmain/pi/dense_3/Tanhmain/pi/add/y*
T0*'
_output_shapes
:         
R
main/pi/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *═╠ї?
`
main/pi/mulMulmain/pi/mul/xmain/pi/add*
T0*'
_output_shapes
:         
T
main/pi/add_1/xConst*
valueB
 *   └*
dtype0*
_output_shapes
: 
d
main/pi/add_1Addmain/pi/add_1/xmain/pi/mul*
T0*'
_output_shapes
:         
S
main/pi/ExpExpmain/pi/add_1*'
_output_shapes
:         *
T0
d
main/pi/ShapeShapemain/pi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
_
main/pi/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
main/pi/random_normal/stddevConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Е
*main/pi/random_normal/RandomStandardNormalRandomStandardNormalmain/pi/Shape*
dtype0*'
_output_shapes
:         *
seed2X*

seed *
T0
ю
main/pi/random_normal/mulMul*main/pi/random_normal/RandomStandardNormalmain/pi/random_normal/stddev*
T0*'
_output_shapes
:         
Ё
main/pi/random_normalAddmain/pi/random_normal/mulmain/pi/random_normal/mean*
T0*'
_output_shapes
:         
j
main/pi/mul_1Mulmain/pi/random_normalmain/pi/Exp*
T0*'
_output_shapes
:         
n
main/pi/add_2Addmain/pi/dense_2/BiasAddmain/pi/mul_1*
T0*'
_output_shapes
:         
_
main/pi/TanhTanhmain/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:         
W
main/pi/Tanh_1Tanhmain/pi/add_2*
T0*'
_output_shapes
:         
n
main/pi/sub_1Submain/pi/add_2main/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:         
U
main/pi/Exp_1Expmain/pi/add_1*'
_output_shapes
:         *
T0
T
main/pi/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *w╠+2
f
main/pi/add_3Addmain/pi/Exp_1main/pi/add_3/y*
T0*'
_output_shapes
:         
j
main/pi/truedivRealDivmain/pi/sub_1main/pi/add_3*
T0*'
_output_shapes
:         
R
main/pi/pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
d
main/pi/powPowmain/pi/truedivmain/pi/pow/y*
T0*'
_output_shapes
:         
T
main/pi/mul_2/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
f
main/pi/mul_2Mulmain/pi/mul_2/xmain/pi/add_1*
T0*'
_output_shapes
:         
b
main/pi/add_4Addmain/pi/powmain/pi/mul_2*'
_output_shapes
:         *
T0
T
main/pi/add_5/yConst*
valueB
 *ј?в?*
dtype0*
_output_shapes
: 
f
main/pi/add_5Addmain/pi/add_4main/pi/add_5/y*
T0*'
_output_shapes
:         
T
main/pi/mul_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ┐
f
main/pi/mul_3Mulmain/pi/mul_3/xmain/pi/add_5*
T0*'
_output_shapes
:         
T
main/pi/pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
g
main/pi/pow_1Powmain/pi/Tanh_1main/pi/pow_1/y*'
_output_shapes
:         *
T0
T
main/pi/sub_2/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
f
main/pi/sub_2Submain/pi/sub_2/xmain/pi/pow_1*
T0*'
_output_shapes
:         
V
main/pi/Greater/yConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
n
main/pi/GreaterGreatermain/pi/sub_2main/pi/Greater/y*
T0*'
_output_shapes
:         
v
main/pi/CastCastmain/pi/Greater*
Truncate( *'
_output_shapes
:         *

DstT0*

SrcT0

S
main/pi/Less/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
e
main/pi/LessLessmain/pi/sub_2main/pi/Less/y*'
_output_shapes
:         *
T0
u
main/pi/Cast_1Castmain/pi/Less*

SrcT0
*
Truncate( *'
_output_shapes
:         *

DstT0
T
main/pi/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
f
main/pi/sub_3Submain/pi/sub_3/xmain/pi/sub_2*
T0*'
_output_shapes
:         
c
main/pi/mul_4Mulmain/pi/sub_3main/pi/Cast*
T0*'
_output_shapes
:         
T
main/pi/sub_4/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
main/pi/sub_4Submain/pi/sub_4/xmain/pi/sub_2*
T0*'
_output_shapes
:         
e
main/pi/mul_5Mulmain/pi/sub_4main/pi/Cast_1*'
_output_shapes
:         *
T0
d
main/pi/add_6Addmain/pi/mul_4main/pi/mul_5*
T0*'
_output_shapes
:         
e
main/pi/StopGradientStopGradientmain/pi/add_6*
T0*'
_output_shapes
:         
k
main/pi/add_7Addmain/pi/sub_2main/pi/StopGradient*'
_output_shapes
:         *
T0
T
main/pi/add_8/yConst*
dtype0*
_output_shapes
: *
valueB
 *й7є5
f
main/pi/add_8Addmain/pi/add_7main/pi/add_8/y*
T0*'
_output_shapes
:         
S
main/pi/LogLogmain/pi/add_8*
T0*'
_output_shapes
:         
b
main/pi/sub_5Submain/pi/mul_3main/pi/Log*
T0*'
_output_shapes
:         
X
main/pi/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
main/pi/Greater_1Greatermain/pi/submain/pi/Greater_1/y*
T0*
_output_shapes
: 
f
main/pi/cond/SwitchSwitchmain/pi/Greater_1main/pi/Greater_1*
_output_shapes
: : *
T0

Y
main/pi/cond/switch_tIdentitymain/pi/cond/Switch:1*
_output_shapes
: *
T0

W
main/pi/cond/switch_fIdentitymain/pi/cond/Switch*
T0
*
_output_shapes
: 
T
main/pi/cond/pred_idIdentitymain/pi/Greater_1*
T0
*
_output_shapes
: 
d
main/pi/cond/ExpExpmain/pi/cond/Exp/Switch:1*
T0*'
_output_shapes
:         
Г
main/pi/cond/Exp/SwitchSwitchmain/pi/sub_5main/pi/cond/pred_id*
T0* 
_class
loc:@main/pi/sub_5*:
_output_shapes(
&:         :         
s
main/pi/cond/Maximum/yConst^main/pi/cond/switch_t*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
{
main/pi/cond/MaximumMaximummain/pi/cond/Expmain/pi/cond/Maximum/y*
T0*'
_output_shapes
:         
q
main/pi/cond/Equal/yConst^main/pi/cond/switch_t*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
o
main/pi/cond/EqualEqualmain/pi/cond/Equal/Switch:1main/pi/cond/Equal/y*
T0*
_output_shapes
: 
Ѕ
main/pi/cond/Equal/SwitchSwitchmain/pi/submain/pi/cond/pred_id*
T0*
_class
loc:@main/pi/sub*
_output_shapes
: : 
m
main/pi/cond/cond/SwitchSwitchmain/pi/cond/Equalmain/pi/cond/Equal*
T0
*
_output_shapes
: : 
c
main/pi/cond/cond/switch_tIdentitymain/pi/cond/cond/Switch:1*
_output_shapes
: *
T0

a
main/pi/cond/cond/switch_fIdentitymain/pi/cond/cond/Switch*
_output_shapes
: *
T0

Z
main/pi/cond/cond/pred_idIdentitymain/pi/cond/Equal*
T0
*
_output_shapes
: 
n
main/pi/cond/cond/LogLogmain/pi/cond/cond/Log/Switch:1*
T0*'
_output_shapes
:         
┼
main/pi/cond/cond/Log/SwitchSwitchmain/pi/cond/Maximummain/pi/cond/cond/pred_id*
T0*'
_class
loc:@main/pi/cond/Maximum*:
_output_shapes(
&:         :         
y
main/pi/cond/cond/sub/xConst^main/pi/cond/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
t
main/pi/cond/cond/subSubmain/pi/cond/cond/sub/xmain/pi/cond/cond/sub/Switch*
T0*
_output_shapes
: 
А
main/pi/cond/cond/sub/SwitchSwitchmain/pi/cond/Equal/Switch:1main/pi/cond/cond/pred_id*
T0*
_class
loc:@main/pi/sub*
_output_shapes
: : 
Ѓ
main/pi/cond/cond/PowPowmain/pi/cond/cond/Pow/Switchmain/pi/cond/cond/sub*
T0*'
_output_shapes
:         
┼
main/pi/cond/cond/Pow/SwitchSwitchmain/pi/cond/Maximummain/pi/cond/cond/pred_id*
T0*'
_class
loc:@main/pi/cond/Maximum*:
_output_shapes(
&:         :         
{
main/pi/cond/cond/sub_1/yConst^main/pi/cond/cond/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ѓ
main/pi/cond/cond/sub_1Submain/pi/cond/cond/Powmain/pi/cond/cond/sub_1/y*
T0*'
_output_shapes
:         
{
main/pi/cond/cond/sub_2/xConst^main/pi/cond/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
x
main/pi/cond/cond/sub_2Submain/pi/cond/cond/sub_2/xmain/pi/cond/cond/sub/Switch*
T0*
_output_shapes
: 
ѕ
main/pi/cond/cond/truedivRealDivmain/pi/cond/cond/sub_1main/pi/cond/cond/sub_2*
T0*'
_output_shapes
:         
Ј
main/pi/cond/cond/MergeMergemain/pi/cond/cond/truedivmain/pi/cond/cond/Log*
N*)
_output_shapes
:         : *
T0
|
"main/pi/cond/Sum/reduction_indicesConst^main/pi/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ъ
main/pi/cond/SumSummain/pi/cond/cond/Merge"main/pi/cond/Sum/reduction_indices*#
_output_shapes
:         *
	keep_dims( *

Tidx0*
T0
f
main/pi/cond/Exp_1Expmain/pi/cond/Exp_1/Switch*'
_output_shapes
:         *
T0
»
main/pi/cond/Exp_1/SwitchSwitchmain/pi/sub_5main/pi/cond/pred_id*
T0* 
_class
loc:@main/pi/sub_5*:
_output_shapes(
&:         :         
o
main/pi/cond/sub/xConst^main/pi/cond/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
e
main/pi/cond/subSubmain/pi/cond/sub/xmain/pi/cond/sub/Switch*
T0*
_output_shapes
: 
Є
main/pi/cond/sub/SwitchSwitchmain/pi/submain/pi/cond/pred_id*
T0*
_class
loc:@main/pi/sub*
_output_shapes
: : 
s
main/pi/cond/truediv/xConst^main/pi/cond/switch_f*
valueB
 *   A*
dtype0*
_output_shapes
: 
j
main/pi/cond/truedivRealDivmain/pi/cond/truediv/xmain/pi/cond/sub*
_output_shapes
: *
T0
o
main/pi/cond/Pow/xConst^main/pi/cond/switch_f*
valueB
 *   A*
dtype0*
_output_shapes
: 
b
main/pi/cond/PowPowmain/pi/cond/Pow/xmain/pi/cond/truediv*
T0*
_output_shapes
: 
w
main/pi/cond/MinimumMinimummain/pi/cond/Exp_1main/pi/cond/Pow*
T0*'
_output_shapes
:         
u
main/pi/cond/Maximum_1/yConst^main/pi/cond/switch_f*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
Ѓ
main/pi/cond/Maximum_1Maximummain/pi/cond/Minimummain/pi/cond/Maximum_1/y*'
_output_shapes
:         *
T0
s
main/pi/cond/Equal_1/yConst^main/pi/cond/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
o
main/pi/cond/Equal_1Equalmain/pi/cond/sub/Switchmain/pi/cond/Equal_1/y*
T0*
_output_shapes
: 
s
main/pi/cond/cond_1/SwitchSwitchmain/pi/cond/Equal_1main/pi/cond/Equal_1*
T0
*
_output_shapes
: : 
g
main/pi/cond/cond_1/switch_tIdentitymain/pi/cond/cond_1/Switch:1*
T0
*
_output_shapes
: 
e
main/pi/cond/cond_1/switch_fIdentitymain/pi/cond/cond_1/Switch*
T0
*
_output_shapes
: 
^
main/pi/cond/cond_1/pred_idIdentitymain/pi/cond/Equal_1*
T0
*
_output_shapes
: 
r
main/pi/cond/cond_1/LogLog main/pi/cond/cond_1/Log/Switch:1*'
_output_shapes
:         *
T0
═
main/pi/cond/cond_1/Log/SwitchSwitchmain/pi/cond/Maximum_1main/pi/cond/cond_1/pred_id*:
_output_shapes(
&:         :         *
T0*)
_class
loc:@main/pi/cond/Maximum_1
}
main/pi/cond/cond_1/sub/xConst^main/pi/cond/cond_1/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
z
main/pi/cond/cond_1/subSubmain/pi/cond/cond_1/sub/xmain/pi/cond/cond_1/sub/Switch*
T0*
_output_shapes
: 
А
main/pi/cond/cond_1/sub/SwitchSwitchmain/pi/cond/sub/Switchmain/pi/cond/cond_1/pred_id*
T0*
_class
loc:@main/pi/sub*
_output_shapes
: : 
Ѕ
main/pi/cond/cond_1/PowPowmain/pi/cond/cond_1/Pow/Switchmain/pi/cond/cond_1/sub*
T0*'
_output_shapes
:         
═
main/pi/cond/cond_1/Pow/SwitchSwitchmain/pi/cond/Maximum_1main/pi/cond/cond_1/pred_id*
T0*)
_class
loc:@main/pi/cond/Maximum_1*:
_output_shapes(
&:         :         

main/pi/cond/cond_1/sub_1/yConst^main/pi/cond/cond_1/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ѕ
main/pi/cond/cond_1/sub_1Submain/pi/cond/cond_1/Powmain/pi/cond/cond_1/sub_1/y*'
_output_shapes
:         *
T0

main/pi/cond/cond_1/sub_2/xConst^main/pi/cond/cond_1/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
~
main/pi/cond/cond_1/sub_2Submain/pi/cond/cond_1/sub_2/xmain/pi/cond/cond_1/sub/Switch*
_output_shapes
: *
T0
ј
main/pi/cond/cond_1/truedivRealDivmain/pi/cond/cond_1/sub_1main/pi/cond/cond_1/sub_2*
T0*'
_output_shapes
:         
Ћ
main/pi/cond/cond_1/MergeMergemain/pi/cond/cond_1/truedivmain/pi/cond/cond_1/Log*
T0*
N*)
_output_shapes
:         : 
~
$main/pi/cond/Sum_1/reduction_indicesConst^main/pi/cond/switch_f*
dtype0*
_output_shapes
: *
value	B :
Ц
main/pi/cond/Sum_1Summain/pi/cond/cond_1/Merge$main/pi/cond/Sum_1/reduction_indices*#
_output_shapes
:         *
	keep_dims( *

Tidx0*
T0
z
main/pi/cond/MergeMergemain/pi/cond/Sum_1main/pi/cond/Sum*
N*%
_output_shapes
:         : *
T0
O

main/mul/yConst*
valueB
 *  ќC*
dtype0*
_output_shapes
: 
[
main/mulMulmain/pi/Tanh
main/mul/y*'
_output_shapes
:         *
T0
Q
main/mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ќC
a

main/mul_1Mulmain/pi/Tanh_1main/mul_1/y*'
_output_shapes
:         *
T0
^
main/q1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
њ
main/q1/concatConcatV2PlaceholderPlaceholder_1main/q1/concat/axis*
T0*
N*'
_output_shapes
:         h*

Tidx0
»
5main/q1/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"h      *'
_class
loc:@main/q1/dense/kernel
А
3main/q1/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *б]Ћй*'
_class
loc:@main/q1/dense/kernel*
dtype0*
_output_shapes
: 
А
3main/q1/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *б]Ћ=*'
_class
loc:@main/q1/dense/kernel*
dtype0*
_output_shapes
: 
 
=main/q1/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5main/q1/dense/kernel/Initializer/random_uniform/shape*
seed2¤*
dtype0*
_output_shapes
:	hђ*

seed *
T0*'
_class
loc:@main/q1/dense/kernel
Ь
3main/q1/dense/kernel/Initializer/random_uniform/subSub3main/q1/dense/kernel/Initializer/random_uniform/max3main/q1/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@main/q1/dense/kernel
Ђ
3main/q1/dense/kernel/Initializer/random_uniform/mulMul=main/q1/dense/kernel/Initializer/random_uniform/RandomUniform3main/q1/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	hђ*
T0*'
_class
loc:@main/q1/dense/kernel
з
/main/q1/dense/kernel/Initializer/random_uniformAdd3main/q1/dense/kernel/Initializer/random_uniform/mul3main/q1/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	hђ
│
main/q1/dense/kernel
VariableV2*
shared_name *'
_class
loc:@main/q1/dense/kernel*
	container *
shape:	hђ*
dtype0*
_output_shapes
:	hђ
У
main/q1/dense/kernel/AssignAssignmain/q1/dense/kernel/main/q1/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
ј
main/q1/dense/kernel/readIdentitymain/q1/dense/kernel*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	hђ
д
4main/q1/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*%
_class
loc:@main/q1/dense/bias*
dtype0*
_output_shapes
:
ќ
*main/q1/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@main/q1/dense/bias*
dtype0*
_output_shapes
: 
ь
$main/q1/dense/bias/Initializer/zerosFill4main/q1/dense/bias/Initializer/zeros/shape_as_tensor*main/q1/dense/bias/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@main/q1/dense/bias*
_output_shapes	
:ђ
Д
main/q1/dense/bias
VariableV2*%
_class
loc:@main/q1/dense/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
М
main/q1/dense/bias/AssignAssignmain/q1/dense/bias$main/q1/dense/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ
ё
main/q1/dense/bias/readIdentitymain/q1/dense/bias*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes	
:ђ
б
main/q1/dense/MatMulMatMulmain/q1/concatmain/q1/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ў
main/q1/dense/BiasAddBiasAddmain/q1/dense/MatMulmain/q1/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
d
main/q1/dense/ReluRelumain/q1/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
│
7main/q1/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *)
_class
loc:@main/q1/dense_1/kernel
Ц
5main/q1/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *  ђй*)
_class
loc:@main/q1/dense_1/kernel
Ц
5main/q1/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  ђ=*)
_class
loc:@main/q1/dense_1/kernel*
dtype0*
_output_shapes
: 
є
?main/q1/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q1/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђђ*

seed *
T0*)
_class
loc:@main/q1/dense_1/kernel*
seed2Р
Ш
5main/q1/dense_1/kernel/Initializer/random_uniform/subSub5main/q1/dense_1/kernel/Initializer/random_uniform/max5main/q1/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@main/q1/dense_1/kernel
і
5main/q1/dense_1/kernel/Initializer/random_uniform/mulMul?main/q1/dense_1/kernel/Initializer/random_uniform/RandomUniform5main/q1/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
ђђ
Ч
1main/q1/dense_1/kernel/Initializer/random_uniformAdd5main/q1/dense_1/kernel/Initializer/random_uniform/mul5main/q1/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
ђђ
╣
main/q1/dense_1/kernel
VariableV2*)
_class
loc:@main/q1/dense_1/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name 
ы
main/q1/dense_1/kernel/AssignAssignmain/q1/dense_1/kernel1main/q1/dense_1/kernel/Initializer/random_uniform*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
Ћ
main/q1/dense_1/kernel/readIdentitymain/q1/dense_1/kernel*
T0*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
ђђ
ъ
&main/q1/dense_1/bias/Initializer/zerosConst*
valueBђ*    *'
_class
loc:@main/q1/dense_1/bias*
dtype0*
_output_shapes	
:ђ
Ф
main/q1/dense_1/bias
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *'
_class
loc:@main/q1/dense_1/bias
█
main/q1/dense_1/bias/AssignAssignmain/q1/dense_1/bias&main/q1/dense_1/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
і
main/q1/dense_1/bias/readIdentitymain/q1/dense_1/bias*
T0*'
_class
loc:@main/q1/dense_1/bias*
_output_shapes	
:ђ
ф
main/q1/dense_1/MatMulMatMulmain/q1/dense/Relumain/q1/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ъ
main/q1/dense_1/BiasAddBiasAddmain/q1/dense_1/MatMulmain/q1/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
h
main/q1/dense_1/ReluRelumain/q1/dense_1/BiasAdd*(
_output_shapes
:         ђ*
T0
│
7main/q1/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@main/q1/dense_2/kernel*
dtype0*
_output_shapes
:
Ц
5main/q1/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *|Пй*)
_class
loc:@main/q1/dense_2/kernel*
dtype0*
_output_shapes
: 
Ц
5main/q1/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *|П=*)
_class
loc:@main/q1/dense_2/kernel*
dtype0*
_output_shapes
: 
Ё
?main/q1/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q1/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0*)
_class
loc:@main/q1/dense_2/kernel*
seed2з
Ш
5main/q1/dense_2/kernel/Initializer/random_uniform/subSub5main/q1/dense_2/kernel/Initializer/random_uniform/max5main/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
: 
Ѕ
5main/q1/dense_2/kernel/Initializer/random_uniform/mulMul?main/q1/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/q1/dense_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
:	ђ
ч
1main/q1/dense_2/kernel/Initializer/random_uniformAdd5main/q1/dense_2/kernel/Initializer/random_uniform/mul5main/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
:	ђ
и
main/q1/dense_2/kernel
VariableV2*
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name *)
_class
loc:@main/q1/dense_2/kernel*
	container 
­
main/q1/dense_2/kernel/AssignAssignmain/q1/dense_2/kernel1main/q1/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel
ћ
main/q1/dense_2/kernel/readIdentitymain/q1/dense_2/kernel*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
:	ђ
ю
&main/q1/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/q1/dense_2/bias*
dtype0*
_output_shapes
:
Е
main/q1/dense_2/bias
VariableV2*'
_class
loc:@main/q1/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
┌
main/q1/dense_2/bias/AssignAssignmain/q1/dense_2/bias&main/q1/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias
Ѕ
main/q1/dense_2/bias/readIdentitymain/q1/dense_2/bias*
T0*'
_class
loc:@main/q1/dense_2/bias*
_output_shapes
:
Ф
main/q1/dense_2/MatMulMatMulmain/q1/dense_1/Relumain/q1/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
ъ
main/q1/dense_2/BiasAddBiasAddmain/q1/dense_2/MatMulmain/q1/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
x
main/q1/SqueezeSqueezemain/q1/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
`
main/q1_1/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
         
Њ
main/q1_1/concatConcatV2Placeholder
main/mul_1main/q1_1/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:         h
д
main/q1_1/dense/MatMulMatMulmain/q1_1/concatmain/q1/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ю
main/q1_1/dense/BiasAddBiasAddmain/q1_1/dense/MatMulmain/q1/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
h
main/q1_1/dense/ReluRelumain/q1_1/dense/BiasAdd*(
_output_shapes
:         ђ*
T0
«
main/q1_1/dense_1/MatMulMatMulmain/q1_1/dense/Relumain/q1/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Б
main/q1_1/dense_1/BiasAddBiasAddmain/q1_1/dense_1/MatMulmain/q1/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
l
main/q1_1/dense_1/ReluRelumain/q1_1/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
»
main/q1_1/dense_2/MatMulMatMulmain/q1_1/dense_1/Relumain/q1/dense_2/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
б
main/q1_1/dense_2/BiasAddBiasAddmain/q1_1/dense_2/MatMulmain/q1/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
|
main/q1_1/SqueezeSqueezemain/q1_1/dense_2/BiasAdd*#
_output_shapes
:         *
squeeze_dims
*
T0
^
main/q2/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
њ
main/q2/concatConcatV2PlaceholderPlaceholder_1main/q2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:         h
»
5main/q2/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"h      *'
_class
loc:@main/q2/dense/kernel*
dtype0*
_output_shapes
:
А
3main/q2/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *б]Ћй*'
_class
loc:@main/q2/dense/kernel
А
3main/q2/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *б]Ћ=*'
_class
loc:@main/q2/dense/kernel
 
=main/q2/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5main/q2/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	hђ*

seed *
T0*'
_class
loc:@main/q2/dense/kernel*
seed2Љ
Ь
3main/q2/dense/kernel/Initializer/random_uniform/subSub3main/q2/dense/kernel/Initializer/random_uniform/max3main/q2/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/q2/dense/kernel*
_output_shapes
: 
Ђ
3main/q2/dense/kernel/Initializer/random_uniform/mulMul=main/q2/dense/kernel/Initializer/random_uniform/RandomUniform3main/q2/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@main/q2/dense/kernel*
_output_shapes
:	hђ
з
/main/q2/dense/kernel/Initializer/random_uniformAdd3main/q2/dense/kernel/Initializer/random_uniform/mul3main/q2/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/q2/dense/kernel*
_output_shapes
:	hђ
│
main/q2/dense/kernel
VariableV2*'
_class
loc:@main/q2/dense/kernel*
	container *
shape:	hђ*
dtype0*
_output_shapes
:	hђ*
shared_name 
У
main/q2/dense/kernel/AssignAssignmain/q2/dense/kernel/main/q2/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	hђ*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel
ј
main/q2/dense/kernel/readIdentitymain/q2/dense/kernel*
T0*'
_class
loc:@main/q2/dense/kernel*
_output_shapes
:	hђ
д
4main/q2/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*%
_class
loc:@main/q2/dense/bias*
dtype0*
_output_shapes
:
ќ
*main/q2/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@main/q2/dense/bias*
dtype0*
_output_shapes
: 
ь
$main/q2/dense/bias/Initializer/zerosFill4main/q2/dense/bias/Initializer/zeros/shape_as_tensor*main/q2/dense/bias/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@main/q2/dense/bias*
_output_shapes	
:ђ
Д
main/q2/dense/bias
VariableV2*
shared_name *%
_class
loc:@main/q2/dense/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
М
main/q2/dense/bias/AssignAssignmain/q2/dense/bias$main/q2/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias
ё
main/q2/dense/bias/readIdentitymain/q2/dense/bias*
_output_shapes	
:ђ*
T0*%
_class
loc:@main/q2/dense/bias
б
main/q2/dense/MatMulMatMulmain/q2/concatmain/q2/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
Ў
main/q2/dense/BiasAddBiasAddmain/q2/dense/MatMulmain/q2/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
d
main/q2/dense/ReluRelumain/q2/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
│
7main/q2/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@main/q2/dense_1/kernel*
dtype0*
_output_shapes
:
Ц
5main/q2/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *  ђй*)
_class
loc:@main/q2/dense_1/kernel*
dtype0*
_output_shapes
: 
Ц
5main/q2/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  ђ=*)
_class
loc:@main/q2/dense_1/kernel*
dtype0*
_output_shapes
: 
є
?main/q2/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q2/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђђ*

seed *
T0*)
_class
loc:@main/q2/dense_1/kernel*
seed2ц
Ш
5main/q2/dense_1/kernel/Initializer/random_uniform/subSub5main/q2/dense_1/kernel/Initializer/random_uniform/max5main/q2/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes
: 
і
5main/q2/dense_1/kernel/Initializer/random_uniform/mulMul?main/q2/dense_1/kernel/Initializer/random_uniform/RandomUniform5main/q2/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:
ђђ
Ч
1main/q2/dense_1/kernel/Initializer/random_uniformAdd5main/q2/dense_1/kernel/Initializer/random_uniform/mul5main/q2/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:
ђђ
╣
main/q2/dense_1/kernel
VariableV2*
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *)
_class
loc:@main/q2/dense_1/kernel*
	container 
ы
main/q2/dense_1/kernel/AssignAssignmain/q2/dense_1/kernel1main/q2/dense_1/kernel/Initializer/random_uniform*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
Ћ
main/q2/dense_1/kernel/readIdentitymain/q2/dense_1/kernel*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:
ђђ
ъ
&main/q2/dense_1/bias/Initializer/zerosConst*
valueBђ*    *'
_class
loc:@main/q2/dense_1/bias*
dtype0*
_output_shapes	
:ђ
Ф
main/q2/dense_1/bias
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *'
_class
loc:@main/q2/dense_1/bias*
	container 
█
main/q2/dense_1/bias/AssignAssignmain/q2/dense_1/bias&main/q2/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias
і
main/q2/dense_1/bias/readIdentitymain/q2/dense_1/bias*
T0*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:ђ
ф
main/q2/dense_1/MatMulMatMulmain/q2/dense/Relumain/q2/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
Ъ
main/q2/dense_1/BiasAddBiasAddmain/q2/dense_1/MatMulmain/q2/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
h
main/q2/dense_1/ReluRelumain/q2/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
│
7main/q2/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@main/q2/dense_2/kernel*
dtype0*
_output_shapes
:
Ц
5main/q2/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *|Пй*)
_class
loc:@main/q2/dense_2/kernel*
dtype0*
_output_shapes
: 
Ц
5main/q2/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *|П=*)
_class
loc:@main/q2/dense_2/kernel*
dtype0*
_output_shapes
: 
Ё
?main/q2/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q2/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0*)
_class
loc:@main/q2/dense_2/kernel*
seed2х
Ш
5main/q2/dense_2/kernel/Initializer/random_uniform/subSub5main/q2/dense_2/kernel/Initializer/random_uniform/max5main/q2/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
: 
Ѕ
5main/q2/dense_2/kernel/Initializer/random_uniform/mulMul?main/q2/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/q2/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/q2/dense_2/kernel
ч
1main/q2/dense_2/kernel/Initializer/random_uniformAdd5main/q2/dense_2/kernel/Initializer/random_uniform/mul5main/q2/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/q2/dense_2/kernel
и
main/q2/dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	ђ*
shared_name *)
_class
loc:@main/q2/dense_2/kernel*
	container *
shape:	ђ
­
main/q2/dense_2/kernel/AssignAssignmain/q2/dense_2/kernel1main/q2/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
ћ
main/q2/dense_2/kernel/readIdentitymain/q2/dense_2/kernel*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/q2/dense_2/kernel
ю
&main/q2/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/q2/dense_2/bias*
dtype0*
_output_shapes
:
Е
main/q2/dense_2/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/q2/dense_2/bias
┌
main/q2/dense_2/bias/AssignAssignmain/q2/dense_2/bias&main/q2/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias
Ѕ
main/q2/dense_2/bias/readIdentitymain/q2/dense_2/bias*
_output_shapes
:*
T0*'
_class
loc:@main/q2/dense_2/bias
Ф
main/q2/dense_2/MatMulMatMulmain/q2/dense_1/Relumain/q2/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
ъ
main/q2/dense_2/BiasAddBiasAddmain/q2/dense_2/MatMulmain/q2/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
x
main/q2/SqueezeSqueezemain/q2/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
`
main/q2_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
Њ
main/q2_1/concatConcatV2Placeholder
main/mul_1main/q2_1/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:         h
д
main/q2_1/dense/MatMulMatMulmain/q2_1/concatmain/q2/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
Ю
main/q2_1/dense/BiasAddBiasAddmain/q2_1/dense/MatMulmain/q2/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
h
main/q2_1/dense/ReluRelumain/q2_1/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
«
main/q2_1/dense_1/MatMulMatMulmain/q2_1/dense/Relumain/q2/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Б
main/q2_1/dense_1/BiasAddBiasAddmain/q2_1/dense_1/MatMulmain/q2/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
l
main/q2_1/dense_1/ReluRelumain/q2_1/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
»
main/q2_1/dense_2/MatMulMatMulmain/q2_1/dense_1/Relumain/q2/dense_2/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
б
main/q2_1/dense_2/BiasAddBiasAddmain/q2_1/dense_2/MatMulmain/q2/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
|
main/q2_1/SqueezeSqueezemain/q2_1/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
Г
4main/v/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"I      *&
_class
loc:@main/v/dense/kernel*
dtype0*
_output_shapes
:
Ъ
2main/v/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *%vЌй*&
_class
loc:@main/v/dense/kernel*
dtype0*
_output_shapes
: 
Ъ
2main/v/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *%vЌ=*&
_class
loc:@main/v/dense/kernel*
dtype0*
_output_shapes
: 
Ч
<main/v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/v/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@main/v/dense/kernel*
seed2Л*
dtype0*
_output_shapes
:	Iђ
Ж
2main/v/dense/kernel/Initializer/random_uniform/subSub2main/v/dense/kernel/Initializer/random_uniform/max2main/v/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*&
_class
loc:@main/v/dense/kernel
§
2main/v/dense/kernel/Initializer/random_uniform/mulMul<main/v/dense/kernel/Initializer/random_uniform/RandomUniform2main/v/dense/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/v/dense/kernel*
_output_shapes
:	Iђ
№
.main/v/dense/kernel/Initializer/random_uniformAdd2main/v/dense/kernel/Initializer/random_uniform/mul2main/v/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/v/dense/kernel*
_output_shapes
:	Iђ
▒
main/v/dense/kernel
VariableV2*
shared_name *&
_class
loc:@main/v/dense/kernel*
	container *
shape:	Iђ*
dtype0*
_output_shapes
:	Iђ
С
main/v/dense/kernel/AssignAssignmain/v/dense/kernel.main/v/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@main/v/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
І
main/v/dense/kernel/readIdentitymain/v/dense/kernel*
T0*&
_class
loc:@main/v/dense/kernel*
_output_shapes
:	Iђ
ц
3main/v/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*$
_class
loc:@main/v/dense/bias*
dtype0*
_output_shapes
:
ћ
)main/v/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@main/v/dense/bias*
dtype0*
_output_shapes
: 
ж
#main/v/dense/bias/Initializer/zerosFill3main/v/dense/bias/Initializer/zeros/shape_as_tensor)main/v/dense/bias/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@main/v/dense/bias*
_output_shapes	
:ђ
Ц
main/v/dense/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *$
_class
loc:@main/v/dense/bias*
	container *
shape:ђ
¤
main/v/dense/bias/AssignAssignmain/v/dense/bias#main/v/dense/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/v/dense/bias*
validate_shape(*
_output_shapes	
:ђ
Ђ
main/v/dense/bias/readIdentitymain/v/dense/bias*
T0*$
_class
loc:@main/v/dense/bias*
_output_shapes	
:ђ
Ю
main/v/dense/MatMulMatMulPlaceholdermain/v/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
ќ
main/v/dense/BiasAddBiasAddmain/v/dense/MatMulmain/v/dense/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
b
main/v/dense/ReluRelumain/v/dense/BiasAdd*(
_output_shapes
:         ђ*
T0
▒
6main/v/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *(
_class
loc:@main/v/dense_1/kernel*
dtype0*
_output_shapes
:
Б
4main/v/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *  ђй*(
_class
loc:@main/v/dense_1/kernel
Б
4main/v/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  ђ=*(
_class
loc:@main/v/dense_1/kernel*
dtype0*
_output_shapes
: 
Ѓ
>main/v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6main/v/dense_1/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@main/v/dense_1/kernel*
seed2С*
dtype0* 
_output_shapes
:
ђђ*

seed 
Ы
4main/v/dense_1/kernel/Initializer/random_uniform/subSub4main/v/dense_1/kernel/Initializer/random_uniform/max4main/v/dense_1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@main/v/dense_1/kernel*
_output_shapes
: 
є
4main/v/dense_1/kernel/Initializer/random_uniform/mulMul>main/v/dense_1/kernel/Initializer/random_uniform/RandomUniform4main/v/dense_1/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@main/v/dense_1/kernel* 
_output_shapes
:
ђђ
Э
0main/v/dense_1/kernel/Initializer/random_uniformAdd4main/v/dense_1/kernel/Initializer/random_uniform/mul4main/v/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
ђђ*
T0*(
_class
loc:@main/v/dense_1/kernel
и
main/v/dense_1/kernel
VariableV2*
shared_name *(
_class
loc:@main/v/dense_1/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ
ь
main/v/dense_1/kernel/AssignAssignmain/v/dense_1/kernel0main/v/dense_1/kernel/Initializer/random_uniform*
T0*(
_class
loc:@main/v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
њ
main/v/dense_1/kernel/readIdentitymain/v/dense_1/kernel*
T0*(
_class
loc:@main/v/dense_1/kernel* 
_output_shapes
:
ђђ
ю
%main/v/dense_1/bias/Initializer/zerosConst*
valueBђ*    *&
_class
loc:@main/v/dense_1/bias*
dtype0*
_output_shapes	
:ђ
Е
main/v/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *&
_class
loc:@main/v/dense_1/bias*
	container *
shape:ђ
О
main/v/dense_1/bias/AssignAssignmain/v/dense_1/bias%main/v/dense_1/bias/Initializer/zeros*
T0*&
_class
loc:@main/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
Є
main/v/dense_1/bias/readIdentitymain/v/dense_1/bias*
T0*&
_class
loc:@main/v/dense_1/bias*
_output_shapes	
:ђ
Д
main/v/dense_1/MatMulMatMulmain/v/dense/Relumain/v/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
ю
main/v/dense_1/BiasAddBiasAddmain/v/dense_1/MatMulmain/v/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
f
main/v/dense_1/ReluRelumain/v/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
▒
6main/v/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *(
_class
loc:@main/v/dense_2/kernel*
dtype0*
_output_shapes
:
Б
4main/v/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *|Пй*(
_class
loc:@main/v/dense_2/kernel*
dtype0*
_output_shapes
: 
Б
4main/v/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *|П=*(
_class
loc:@main/v/dense_2/kernel*
dtype0*
_output_shapes
: 
ѓ
>main/v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6main/v/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0*(
_class
loc:@main/v/dense_2/kernel*
seed2ш
Ы
4main/v/dense_2/kernel/Initializer/random_uniform/subSub4main/v/dense_2/kernel/Initializer/random_uniform/max4main/v/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@main/v/dense_2/kernel*
_output_shapes
: 
Ё
4main/v/dense_2/kernel/Initializer/random_uniform/mulMul>main/v/dense_2/kernel/Initializer/random_uniform/RandomUniform4main/v/dense_2/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@main/v/dense_2/kernel*
_output_shapes
:	ђ
э
0main/v/dense_2/kernel/Initializer/random_uniformAdd4main/v/dense_2/kernel/Initializer/random_uniform/mul4main/v/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@main/v/dense_2/kernel*
_output_shapes
:	ђ
х
main/v/dense_2/kernel
VariableV2*
shared_name *(
_class
loc:@main/v/dense_2/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
В
main/v/dense_2/kernel/AssignAssignmain/v/dense_2/kernel0main/v/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*(
_class
loc:@main/v/dense_2/kernel
Љ
main/v/dense_2/kernel/readIdentitymain/v/dense_2/kernel*
_output_shapes
:	ђ*
T0*(
_class
loc:@main/v/dense_2/kernel
џ
%main/v/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *&
_class
loc:@main/v/dense_2/bias
Д
main/v/dense_2/bias
VariableV2*
shared_name *&
_class
loc:@main/v/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
о
main/v/dense_2/bias/AssignAssignmain/v/dense_2/bias%main/v/dense_2/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/v/dense_2/bias*
validate_shape(*
_output_shapes
:
є
main/v/dense_2/bias/readIdentitymain/v/dense_2/bias*
T0*&
_class
loc:@main/v/dense_2/bias*
_output_shapes
:
е
main/v/dense_2/MatMulMatMulmain/v/dense_1/Relumain/v/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
Џ
main/v/dense_2/BiasAddBiasAddmain/v/dense_2/MatMulmain/v/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
v
main/v/SqueezeSqueezemain/v/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
T
target/pi/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
U
target/pi/subSubtarget/pi/sub/xPlaceholder_6*
T0*
_output_shapes
: 
│
7target/pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"I      *)
_class
loc:@target/pi/dense/kernel*
dtype0*
_output_shapes
:
Ц
5target/pi/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *%vЌй*)
_class
loc:@target/pi/dense/kernel*
dtype0*
_output_shapes
: 
Ц
5target/pi/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *%vЌ=*)
_class
loc:@target/pi/dense/kernel
Ё
?target/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7target/pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	Iђ*

seed *
T0*)
_class
loc:@target/pi/dense/kernel*
seed2ѕ
Ш
5target/pi/dense/kernel/Initializer/random_uniform/subSub5target/pi/dense/kernel/Initializer/random_uniform/max5target/pi/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
: 
Ѕ
5target/pi/dense/kernel/Initializer/random_uniform/mulMul?target/pi/dense/kernel/Initializer/random_uniform/RandomUniform5target/pi/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	Iђ
ч
1target/pi/dense/kernel/Initializer/random_uniformAdd5target/pi/dense/kernel/Initializer/random_uniform/mul5target/pi/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	Iђ
и
target/pi/dense/kernel
VariableV2*
shared_name *)
_class
loc:@target/pi/dense/kernel*
	container *
shape:	Iђ*
dtype0*
_output_shapes
:	Iђ
­
target/pi/dense/kernel/AssignAssigntarget/pi/dense/kernel1target/pi/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
ћ
target/pi/dense/kernel/readIdentitytarget/pi/dense/kernel*
_output_shapes
:	Iђ*
T0*)
_class
loc:@target/pi/dense/kernel
ф
6target/pi/dense/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:ђ*'
_class
loc:@target/pi/dense/bias
џ
,target/pi/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@target/pi/dense/bias*
dtype0*
_output_shapes
: 
ш
&target/pi/dense/bias/Initializer/zerosFill6target/pi/dense/bias/Initializer/zeros/shape_as_tensor,target/pi/dense/bias/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@target/pi/dense/bias*
_output_shapes	
:ђ
Ф
target/pi/dense/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *'
_class
loc:@target/pi/dense/bias*
	container *
shape:ђ
█
target/pi/dense/bias/AssignAssigntarget/pi/dense/bias&target/pi/dense/bias/Initializer/zeros*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
і
target/pi/dense/bias/readIdentitytarget/pi/dense/bias*
T0*'
_class
loc:@target/pi/dense/bias*
_output_shapes	
:ђ
Ц
target/pi/dense/MatMulMatMulPlaceholder_2target/pi/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ъ
target/pi/dense/BiasAddBiasAddtarget/pi/dense/MatMultarget/pi/dense/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
h
target/pi/dense/ReluRelutarget/pi/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
и
9target/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *+
_class!
loc:@target/pi/dense_1/kernel
Е
7target/pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *  ђй*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
Е
7target/pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  ђ=*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
ї
Atarget/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђђ*

seed *
T0*+
_class!
loc:@target/pi/dense_1/kernel*
seed2Џ
■
7target/pi/dense_1/kernel/Initializer/random_uniform/subSub7target/pi/dense_1/kernel/Initializer/random_uniform/max7target/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
_output_shapes
: 
њ
7target/pi/dense_1/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_1/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
ђђ
ё
3target/pi/dense_1/kernel/Initializer/random_uniformAdd7target/pi/dense_1/kernel/Initializer/random_uniform/mul7target/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
ђђ
й
target/pi/dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *+
_class!
loc:@target/pi/dense_1/kernel*
	container *
shape:
ђђ
щ
target/pi/dense_1/kernel/AssignAssigntarget/pi/dense_1/kernel3target/pi/dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel
Џ
target/pi/dense_1/kernel/readIdentitytarget/pi/dense_1/kernel* 
_output_shapes
:
ђђ*
T0*+
_class!
loc:@target/pi/dense_1/kernel
б
(target/pi/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*
valueBђ*    *)
_class
loc:@target/pi/dense_1/bias
»
target/pi/dense_1/bias
VariableV2*)
_class
loc:@target/pi/dense_1/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
с
target/pi/dense_1/bias/AssignAssigntarget/pi/dense_1/bias(target/pi/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias
љ
target/pi/dense_1/bias/readIdentitytarget/pi/dense_1/bias*
T0*)
_class
loc:@target/pi/dense_1/bias*
_output_shapes	
:ђ
░
target/pi/dense_1/MatMulMatMultarget/pi/dense/Relutarget/pi/dense_1/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( *
T0
Ц
target/pi/dense_1/BiasAddBiasAddtarget/pi/dense_1/MatMultarget/pi/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
l
target/pi/dense_1/ReluRelutarget/pi/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
и
9target/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@target/pi/dense_2/kernel*
dtype0*
_output_shapes
:
Е
7target/pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *ЬGОй*+
_class!
loc:@target/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
Е
7target/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ЬGО=*+
_class!
loc:@target/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
І
Atarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0*+
_class!
loc:@target/pi/dense_2/kernel*
seed2г
■
7target/pi/dense_2/kernel/Initializer/random_uniform/subSub7target/pi/dense_2/kernel/Initializer/random_uniform/max7target/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
: 
Љ
7target/pi/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	ђ
Ѓ
3target/pi/dense_2/kernel/Initializer/random_uniformAdd7target/pi/dense_2/kernel/Initializer/random_uniform/mul7target/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	ђ
╗
target/pi/dense_2/kernel
VariableV2*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name *+
_class!
loc:@target/pi/dense_2/kernel
Э
target/pi/dense_2/kernel/AssignAssigntarget/pi/dense_2/kernel3target/pi/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
џ
target/pi/dense_2/kernel/readIdentitytarget/pi/dense_2/kernel*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	ђ
а
(target/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/pi/dense_2/bias*
dtype0*
_output_shapes
:
Г
target/pi/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@target/pi/dense_2/bias*
	container *
shape:
Р
target/pi/dense_2/bias/AssignAssigntarget/pi/dense_2/bias(target/pi/dense_2/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
Ј
target/pi/dense_2/bias/readIdentitytarget/pi/dense_2/bias*
T0*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:
▒
target/pi/dense_2/MatMulMatMultarget/pi/dense_1/Relutarget/pi/dense_2/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
ц
target/pi/dense_2/BiasAddBiasAddtarget/pi/dense_2/MatMultarget/pi/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
и
9target/pi/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@target/pi/dense_3/kernel*
dtype0*
_output_shapes
:
Е
7target/pi/dense_3/kernel/Initializer/random_uniform/minConst*
valueB
 *ЬGОй*+
_class!
loc:@target/pi/dense_3/kernel*
dtype0*
_output_shapes
: 
Е
7target/pi/dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *ЬGО=*+
_class!
loc:@target/pi/dense_3/kernel*
dtype0*
_output_shapes
: 
І
Atarget/pi/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0*+
_class!
loc:@target/pi/dense_3/kernel*
seed2╝
■
7target/pi/dense_3/kernel/Initializer/random_uniform/subSub7target/pi/dense_3/kernel/Initializer/random_uniform/max7target/pi/dense_3/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_3/kernel*
_output_shapes
: 
Љ
7target/pi/dense_3/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_3/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes
:	ђ*
T0*+
_class!
loc:@target/pi/dense_3/kernel
Ѓ
3target/pi/dense_3/kernel/Initializer/random_uniformAdd7target/pi/dense_3/kernel/Initializer/random_uniform/mul7target/pi/dense_3/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_3/kernel*
_output_shapes
:	ђ
╗
target/pi/dense_3/kernel
VariableV2*
shared_name *+
_class!
loc:@target/pi/dense_3/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
Э
target/pi/dense_3/kernel/AssignAssigntarget/pi/dense_3/kernel3target/pi/dense_3/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ
џ
target/pi/dense_3/kernel/readIdentitytarget/pi/dense_3/kernel*
T0*+
_class!
loc:@target/pi/dense_3/kernel*
_output_shapes
:	ђ
а
(target/pi/dense_3/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/pi/dense_3/bias*
dtype0*
_output_shapes
:
Г
target/pi/dense_3/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@target/pi/dense_3/bias*
	container *
shape:
Р
target/pi/dense_3/bias/AssignAssigntarget/pi/dense_3/bias(target/pi/dense_3/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@target/pi/dense_3/bias*
validate_shape(*
_output_shapes
:
Ј
target/pi/dense_3/bias/readIdentitytarget/pi/dense_3/bias*
T0*)
_class
loc:@target/pi/dense_3/bias*
_output_shapes
:
▒
target/pi/dense_3/MatMulMatMultarget/pi/dense_1/Relutarget/pi/dense_3/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
ц
target/pi/dense_3/BiasAddBiasAddtarget/pi/dense_3/MatMultarget/pi/dense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
k
target/pi/dense_3/TanhTanhtarget/pi/dense_3/BiasAdd*'
_output_shapes
:         *
T0
T
target/pi/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
o
target/pi/addAddtarget/pi/dense_3/Tanhtarget/pi/add/y*
T0*'
_output_shapes
:         
T
target/pi/mul/xConst*
valueB
 *═╠ї?*
dtype0*
_output_shapes
: 
f
target/pi/mulMultarget/pi/mul/xtarget/pi/add*'
_output_shapes
:         *
T0
V
target/pi/add_1/xConst*
valueB
 *   └*
dtype0*
_output_shapes
: 
j
target/pi/add_1Addtarget/pi/add_1/xtarget/pi/mul*
T0*'
_output_shapes
:         
W
target/pi/ExpExptarget/pi/add_1*
T0*'
_output_shapes
:         
h
target/pi/ShapeShapetarget/pi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
a
target/pi/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
c
target/pi/random_normal/stddevConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
«
,target/pi/random_normal/RandomStandardNormalRandomStandardNormaltarget/pi/Shape*
dtype0*'
_output_shapes
:         *
seed2н*

seed *
T0
б
target/pi/random_normal/mulMul,target/pi/random_normal/RandomStandardNormaltarget/pi/random_normal/stddev*
T0*'
_output_shapes
:         
І
target/pi/random_normalAddtarget/pi/random_normal/multarget/pi/random_normal/mean*
T0*'
_output_shapes
:         
p
target/pi/mul_1Multarget/pi/random_normaltarget/pi/Exp*'
_output_shapes
:         *
T0
t
target/pi/add_2Addtarget/pi/dense_2/BiasAddtarget/pi/mul_1*
T0*'
_output_shapes
:         
c
target/pi/TanhTanhtarget/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:         
[
target/pi/Tanh_1Tanhtarget/pi/add_2*'
_output_shapes
:         *
T0
t
target/pi/sub_1Subtarget/pi/add_2target/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:         
Y
target/pi/Exp_1Exptarget/pi/add_1*
T0*'
_output_shapes
:         
V
target/pi/add_3/yConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
l
target/pi/add_3Addtarget/pi/Exp_1target/pi/add_3/y*
T0*'
_output_shapes
:         
p
target/pi/truedivRealDivtarget/pi/sub_1target/pi/add_3*
T0*'
_output_shapes
:         
T
target/pi/pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
j
target/pi/powPowtarget/pi/truedivtarget/pi/pow/y*
T0*'
_output_shapes
:         
V
target/pi/mul_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
l
target/pi/mul_2Multarget/pi/mul_2/xtarget/pi/add_1*'
_output_shapes
:         *
T0
h
target/pi/add_4Addtarget/pi/powtarget/pi/mul_2*
T0*'
_output_shapes
:         
V
target/pi/add_5/yConst*
valueB
 *ј?в?*
dtype0*
_output_shapes
: 
l
target/pi/add_5Addtarget/pi/add_4target/pi/add_5/y*
T0*'
_output_shapes
:         
V
target/pi/mul_3/xConst*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
l
target/pi/mul_3Multarget/pi/mul_3/xtarget/pi/add_5*
T0*'
_output_shapes
:         
V
target/pi/pow_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
m
target/pi/pow_1Powtarget/pi/Tanh_1target/pi/pow_1/y*
T0*'
_output_shapes
:         
V
target/pi/sub_2/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
l
target/pi/sub_2Subtarget/pi/sub_2/xtarget/pi/pow_1*
T0*'
_output_shapes
:         
X
target/pi/Greater/yConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
t
target/pi/GreaterGreatertarget/pi/sub_2target/pi/Greater/y*
T0*'
_output_shapes
:         
z
target/pi/CastCasttarget/pi/Greater*

SrcT0
*
Truncate( *'
_output_shapes
:         *

DstT0
U
target/pi/Less/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
target/pi/LessLesstarget/pi/sub_2target/pi/Less/y*
T0*'
_output_shapes
:         
y
target/pi/Cast_1Casttarget/pi/Less*

SrcT0
*
Truncate( *'
_output_shapes
:         *

DstT0
V
target/pi/sub_3/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
l
target/pi/sub_3Subtarget/pi/sub_3/xtarget/pi/sub_2*'
_output_shapes
:         *
T0
i
target/pi/mul_4Multarget/pi/sub_3target/pi/Cast*
T0*'
_output_shapes
:         
V
target/pi/sub_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
l
target/pi/sub_4Subtarget/pi/sub_4/xtarget/pi/sub_2*
T0*'
_output_shapes
:         
k
target/pi/mul_5Multarget/pi/sub_4target/pi/Cast_1*'
_output_shapes
:         *
T0
j
target/pi/add_6Addtarget/pi/mul_4target/pi/mul_5*
T0*'
_output_shapes
:         
i
target/pi/StopGradientStopGradienttarget/pi/add_6*
T0*'
_output_shapes
:         
q
target/pi/add_7Addtarget/pi/sub_2target/pi/StopGradient*
T0*'
_output_shapes
:         
V
target/pi/add_8/yConst*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
l
target/pi/add_8Addtarget/pi/add_7target/pi/add_8/y*
T0*'
_output_shapes
:         
W
target/pi/LogLogtarget/pi/add_8*
T0*'
_output_shapes
:         
h
target/pi/sub_5Subtarget/pi/mul_3target/pi/Log*
T0*'
_output_shapes
:         
Z
target/pi/Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
e
target/pi/Greater_1Greatertarget/pi/subtarget/pi/Greater_1/y*
_output_shapes
: *
T0
l
target/pi/cond/SwitchSwitchtarget/pi/Greater_1target/pi/Greater_1*
T0
*
_output_shapes
: : 
]
target/pi/cond/switch_tIdentitytarget/pi/cond/Switch:1*
T0
*
_output_shapes
: 
[
target/pi/cond/switch_fIdentitytarget/pi/cond/Switch*
T0
*
_output_shapes
: 
X
target/pi/cond/pred_idIdentitytarget/pi/Greater_1*
T0
*
_output_shapes
: 
h
target/pi/cond/ExpExptarget/pi/cond/Exp/Switch:1*'
_output_shapes
:         *
T0
х
target/pi/cond/Exp/SwitchSwitchtarget/pi/sub_5target/pi/cond/pred_id*
T0*"
_class
loc:@target/pi/sub_5*:
_output_shapes(
&:         :         
w
target/pi/cond/Maximum/yConst^target/pi/cond/switch_t*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
Ђ
target/pi/cond/MaximumMaximumtarget/pi/cond/Exptarget/pi/cond/Maximum/y*'
_output_shapes
:         *
T0
u
target/pi/cond/Equal/yConst^target/pi/cond/switch_t*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
u
target/pi/cond/EqualEqualtarget/pi/cond/Equal/Switch:1target/pi/cond/Equal/y*
T0*
_output_shapes
: 
Љ
target/pi/cond/Equal/SwitchSwitchtarget/pi/subtarget/pi/cond/pred_id*
_output_shapes
: : *
T0* 
_class
loc:@target/pi/sub
s
target/pi/cond/cond/SwitchSwitchtarget/pi/cond/Equaltarget/pi/cond/Equal*
_output_shapes
: : *
T0

g
target/pi/cond/cond/switch_tIdentitytarget/pi/cond/cond/Switch:1*
T0
*
_output_shapes
: 
e
target/pi/cond/cond/switch_fIdentitytarget/pi/cond/cond/Switch*
T0
*
_output_shapes
: 
^
target/pi/cond/cond/pred_idIdentitytarget/pi/cond/Equal*
T0
*
_output_shapes
: 
r
target/pi/cond/cond/LogLog target/pi/cond/cond/Log/Switch:1*
T0*'
_output_shapes
:         
═
target/pi/cond/cond/Log/SwitchSwitchtarget/pi/cond/Maximumtarget/pi/cond/cond/pred_id*:
_output_shapes(
&:         :         *
T0*)
_class
loc:@target/pi/cond/Maximum
}
target/pi/cond/cond/sub/xConst^target/pi/cond/cond/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
z
target/pi/cond/cond/subSubtarget/pi/cond/cond/sub/xtarget/pi/cond/cond/sub/Switch*
_output_shapes
: *
T0
Е
target/pi/cond/cond/sub/SwitchSwitchtarget/pi/cond/Equal/Switch:1target/pi/cond/cond/pred_id*
T0* 
_class
loc:@target/pi/sub*
_output_shapes
: : 
Ѕ
target/pi/cond/cond/PowPowtarget/pi/cond/cond/Pow/Switchtarget/pi/cond/cond/sub*'
_output_shapes
:         *
T0
═
target/pi/cond/cond/Pow/SwitchSwitchtarget/pi/cond/Maximumtarget/pi/cond/cond/pred_id*
T0*)
_class
loc:@target/pi/cond/Maximum*:
_output_shapes(
&:         :         

target/pi/cond/cond/sub_1/yConst^target/pi/cond/cond/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ѕ
target/pi/cond/cond/sub_1Subtarget/pi/cond/cond/Powtarget/pi/cond/cond/sub_1/y*'
_output_shapes
:         *
T0

target/pi/cond/cond/sub_2/xConst^target/pi/cond/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
~
target/pi/cond/cond/sub_2Subtarget/pi/cond/cond/sub_2/xtarget/pi/cond/cond/sub/Switch*
T0*
_output_shapes
: 
ј
target/pi/cond/cond/truedivRealDivtarget/pi/cond/cond/sub_1target/pi/cond/cond/sub_2*
T0*'
_output_shapes
:         
Ћ
target/pi/cond/cond/MergeMergetarget/pi/cond/cond/truedivtarget/pi/cond/cond/Log*
T0*
N*)
_output_shapes
:         : 
ђ
$target/pi/cond/Sum/reduction_indicesConst^target/pi/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ц
target/pi/cond/SumSumtarget/pi/cond/cond/Merge$target/pi/cond/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:         
j
target/pi/cond/Exp_1Exptarget/pi/cond/Exp_1/Switch*'
_output_shapes
:         *
T0
и
target/pi/cond/Exp_1/SwitchSwitchtarget/pi/sub_5target/pi/cond/pred_id*
T0*"
_class
loc:@target/pi/sub_5*:
_output_shapes(
&:         :         
s
target/pi/cond/sub/xConst^target/pi/cond/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
k
target/pi/cond/subSubtarget/pi/cond/sub/xtarget/pi/cond/sub/Switch*
T0*
_output_shapes
: 
Ј
target/pi/cond/sub/SwitchSwitchtarget/pi/subtarget/pi/cond/pred_id*
T0* 
_class
loc:@target/pi/sub*
_output_shapes
: : 
w
target/pi/cond/truediv/xConst^target/pi/cond/switch_f*
valueB
 *   A*
dtype0*
_output_shapes
: 
p
target/pi/cond/truedivRealDivtarget/pi/cond/truediv/xtarget/pi/cond/sub*
T0*
_output_shapes
: 
s
target/pi/cond/Pow/xConst^target/pi/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *   A
h
target/pi/cond/PowPowtarget/pi/cond/Pow/xtarget/pi/cond/truediv*
T0*
_output_shapes
: 
}
target/pi/cond/MinimumMinimumtarget/pi/cond/Exp_1target/pi/cond/Pow*
T0*'
_output_shapes
:         
y
target/pi/cond/Maximum_1/yConst^target/pi/cond/switch_f*
valueB
 *й7є5*
dtype0*
_output_shapes
: 
Ѕ
target/pi/cond/Maximum_1Maximumtarget/pi/cond/Minimumtarget/pi/cond/Maximum_1/y*
T0*'
_output_shapes
:         
w
target/pi/cond/Equal_1/yConst^target/pi/cond/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
u
target/pi/cond/Equal_1Equaltarget/pi/cond/sub/Switchtarget/pi/cond/Equal_1/y*
_output_shapes
: *
T0
y
target/pi/cond/cond_1/SwitchSwitchtarget/pi/cond/Equal_1target/pi/cond/Equal_1*
_output_shapes
: : *
T0

k
target/pi/cond/cond_1/switch_tIdentitytarget/pi/cond/cond_1/Switch:1*
T0
*
_output_shapes
: 
i
target/pi/cond/cond_1/switch_fIdentitytarget/pi/cond/cond_1/Switch*
_output_shapes
: *
T0

b
target/pi/cond/cond_1/pred_idIdentitytarget/pi/cond/Equal_1*
_output_shapes
: *
T0

v
target/pi/cond/cond_1/LogLog"target/pi/cond/cond_1/Log/Switch:1*'
_output_shapes
:         *
T0
Н
 target/pi/cond/cond_1/Log/SwitchSwitchtarget/pi/cond/Maximum_1target/pi/cond/cond_1/pred_id*
T0*+
_class!
loc:@target/pi/cond/Maximum_1*:
_output_shapes(
&:         :         
Ђ
target/pi/cond/cond_1/sub/xConst^target/pi/cond/cond_1/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ђ
target/pi/cond/cond_1/subSubtarget/pi/cond/cond_1/sub/x target/pi/cond/cond_1/sub/Switch*
_output_shapes
: *
T0
Е
 target/pi/cond/cond_1/sub/SwitchSwitchtarget/pi/cond/sub/Switchtarget/pi/cond/cond_1/pred_id*
T0* 
_class
loc:@target/pi/sub*
_output_shapes
: : 
Ј
target/pi/cond/cond_1/PowPow target/pi/cond/cond_1/Pow/Switchtarget/pi/cond/cond_1/sub*
T0*'
_output_shapes
:         
Н
 target/pi/cond/cond_1/Pow/SwitchSwitchtarget/pi/cond/Maximum_1target/pi/cond/cond_1/pred_id*:
_output_shapes(
&:         :         *
T0*+
_class!
loc:@target/pi/cond/Maximum_1
Ѓ
target/pi/cond/cond_1/sub_1/yConst^target/pi/cond/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
ј
target/pi/cond/cond_1/sub_1Subtarget/pi/cond/cond_1/Powtarget/pi/cond/cond_1/sub_1/y*
T0*'
_output_shapes
:         
Ѓ
target/pi/cond/cond_1/sub_2/xConst^target/pi/cond/cond_1/switch_f*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ё
target/pi/cond/cond_1/sub_2Subtarget/pi/cond/cond_1/sub_2/x target/pi/cond/cond_1/sub/Switch*
_output_shapes
: *
T0
ћ
target/pi/cond/cond_1/truedivRealDivtarget/pi/cond/cond_1/sub_1target/pi/cond/cond_1/sub_2*'
_output_shapes
:         *
T0
Џ
target/pi/cond/cond_1/MergeMergetarget/pi/cond/cond_1/truedivtarget/pi/cond/cond_1/Log*
T0*
N*)
_output_shapes
:         : 
ѓ
&target/pi/cond/Sum_1/reduction_indicesConst^target/pi/cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
Ф
target/pi/cond/Sum_1Sumtarget/pi/cond/cond_1/Merge&target/pi/cond/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:         
ђ
target/pi/cond/MergeMergetarget/pi/cond/Sum_1target/pi/cond/Sum*
N*%
_output_shapes
:         : *
T0
Q
target/mul/yConst*
valueB
 *  ќC*
dtype0*
_output_shapes
: 
a

target/mulMultarget/pi/Tanhtarget/mul/y*
T0*'
_output_shapes
:         
S
target/mul_1/yConst*
valueB
 *  ќC*
dtype0*
_output_shapes
: 
g
target/mul_1Multarget/pi/Tanh_1target/mul_1/y*
T0*'
_output_shapes
:         
`
target/q1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
ў
target/q1/concatConcatV2Placeholder_2Placeholder_1target/q1/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:         h
│
7target/q1/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"h      *)
_class
loc:@target/q1/dense/kernel*
dtype0*
_output_shapes
:
Ц
5target/q1/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *б]Ћй*)
_class
loc:@target/q1/dense/kernel*
dtype0*
_output_shapes
: 
Ц
5target/q1/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *б]Ћ=*)
_class
loc:@target/q1/dense/kernel*
dtype0*
_output_shapes
: 
Ё
?target/q1/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7target/q1/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@target/q1/dense/kernel*
seed2╦*
dtype0*
_output_shapes
:	hђ
Ш
5target/q1/dense/kernel/Initializer/random_uniform/subSub5target/q1/dense/kernel/Initializer/random_uniform/max5target/q1/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@target/q1/dense/kernel
Ѕ
5target/q1/dense/kernel/Initializer/random_uniform/mulMul?target/q1/dense/kernel/Initializer/random_uniform/RandomUniform5target/q1/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@target/q1/dense/kernel*
_output_shapes
:	hђ
ч
1target/q1/dense/kernel/Initializer/random_uniformAdd5target/q1/dense/kernel/Initializer/random_uniform/mul5target/q1/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/q1/dense/kernel*
_output_shapes
:	hђ
и
target/q1/dense/kernel
VariableV2*)
_class
loc:@target/q1/dense/kernel*
	container *
shape:	hђ*
dtype0*
_output_shapes
:	hђ*
shared_name 
­
target/q1/dense/kernel/AssignAssigntarget/q1/dense/kernel1target/q1/dense/kernel/Initializer/random_uniform*
T0*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
_output_shapes
:	hђ*
use_locking(
ћ
target/q1/dense/kernel/readIdentitytarget/q1/dense/kernel*
_output_shapes
:	hђ*
T0*)
_class
loc:@target/q1/dense/kernel
ф
6target/q1/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*'
_class
loc:@target/q1/dense/bias*
dtype0*
_output_shapes
:
џ
,target/q1/dense/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *'
_class
loc:@target/q1/dense/bias
ш
&target/q1/dense/bias/Initializer/zerosFill6target/q1/dense/bias/Initializer/zeros/shape_as_tensor,target/q1/dense/bias/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@target/q1/dense/bias*
_output_shapes	
:ђ
Ф
target/q1/dense/bias
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *'
_class
loc:@target/q1/dense/bias*
	container 
█
target/q1/dense/bias/AssignAssigntarget/q1/dense/bias&target/q1/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ
і
target/q1/dense/bias/readIdentitytarget/q1/dense/bias*
T0*'
_class
loc:@target/q1/dense/bias*
_output_shapes	
:ђ
е
target/q1/dense/MatMulMatMultarget/q1/concattarget/q1/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ъ
target/q1/dense/BiasAddBiasAddtarget/q1/dense/MatMultarget/q1/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
h
target/q1/dense/ReluRelutarget/q1/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
и
9target/q1/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@target/q1/dense_1/kernel*
dtype0*
_output_shapes
:
Е
7target/q1/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *  ђй*+
_class!
loc:@target/q1/dense_1/kernel
Е
7target/q1/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  ђ=*+
_class!
loc:@target/q1/dense_1/kernel*
dtype0*
_output_shapes
: 
ї
Atarget/q1/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q1/dense_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
seed2я*
dtype0* 
_output_shapes
:
ђђ*

seed 
■
7target/q1/dense_1/kernel/Initializer/random_uniform/subSub7target/q1/dense_1/kernel/Initializer/random_uniform/max7target/q1/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
_output_shapes
: 
њ
7target/q1/dense_1/kernel/Initializer/random_uniform/mulMulAtarget/q1/dense_1/kernel/Initializer/random_uniform/RandomUniform7target/q1/dense_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/q1/dense_1/kernel* 
_output_shapes
:
ђђ
ё
3target/q1/dense_1/kernel/Initializer/random_uniformAdd7target/q1/dense_1/kernel/Initializer/random_uniform/mul7target/q1/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q1/dense_1/kernel* 
_output_shapes
:
ђђ
й
target/q1/dense_1/kernel
VariableV2*
shared_name *+
_class!
loc:@target/q1/dense_1/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ
щ
target/q1/dense_1/kernel/AssignAssigntarget/q1/dense_1/kernel3target/q1/dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel
Џ
target/q1/dense_1/kernel/readIdentitytarget/q1/dense_1/kernel* 
_output_shapes
:
ђђ*
T0*+
_class!
loc:@target/q1/dense_1/kernel
б
(target/q1/dense_1/bias/Initializer/zerosConst*
valueBђ*    *)
_class
loc:@target/q1/dense_1/bias*
dtype0*
_output_shapes	
:ђ
»
target/q1/dense_1/bias
VariableV2*)
_class
loc:@target/q1/dense_1/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
с
target/q1/dense_1/bias/AssignAssigntarget/q1/dense_1/bias(target/q1/dense_1/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
љ
target/q1/dense_1/bias/readIdentitytarget/q1/dense_1/bias*
T0*)
_class
loc:@target/q1/dense_1/bias*
_output_shapes	
:ђ
░
target/q1/dense_1/MatMulMatMultarget/q1/dense/Relutarget/q1/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ц
target/q1/dense_1/BiasAddBiasAddtarget/q1/dense_1/MatMultarget/q1/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
l
target/q1/dense_1/ReluRelutarget/q1/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
и
9target/q1/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@target/q1/dense_2/kernel*
dtype0*
_output_shapes
:
Е
7target/q1/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *|Пй*+
_class!
loc:@target/q1/dense_2/kernel*
dtype0*
_output_shapes
: 
Е
7target/q1/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *|П=*+
_class!
loc:@target/q1/dense_2/kernel*
dtype0*
_output_shapes
: 
І
Atarget/q1/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q1/dense_2/kernel/Initializer/random_uniform/shape*
seed2№*
dtype0*
_output_shapes
:	ђ*

seed *
T0*+
_class!
loc:@target/q1/dense_2/kernel
■
7target/q1/dense_2/kernel/Initializer/random_uniform/subSub7target/q1/dense_2/kernel/Initializer/random_uniform/max7target/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes
: 
Љ
7target/q1/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/q1/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/q1/dense_2/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes
:	ђ
Ѓ
3target/q1/dense_2/kernel/Initializer/random_uniformAdd7target/q1/dense_2/kernel/Initializer/random_uniform/mul7target/q1/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	ђ*
T0*+
_class!
loc:@target/q1/dense_2/kernel
╗
target/q1/dense_2/kernel
VariableV2*
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name *+
_class!
loc:@target/q1/dense_2/kernel*
	container 
Э
target/q1/dense_2/kernel/AssignAssigntarget/q1/dense_2/kernel3target/q1/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_2/kernel
џ
target/q1/dense_2/kernel/readIdentitytarget/q1/dense_2/kernel*
_output_shapes
:	ђ*
T0*+
_class!
loc:@target/q1/dense_2/kernel
а
(target/q1/dense_2/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/q1/dense_2/bias*
dtype0*
_output_shapes
:
Г
target/q1/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@target/q1/dense_2/bias*
	container *
shape:
Р
target/q1/dense_2/bias/AssignAssigntarget/q1/dense_2/bias(target/q1/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/q1/dense_2/bias
Ј
target/q1/dense_2/bias/readIdentitytarget/q1/dense_2/bias*
T0*)
_class
loc:@target/q1/dense_2/bias*
_output_shapes
:
▒
target/q1/dense_2/MatMulMatMultarget/q1/dense_1/Relutarget/q1/dense_2/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
ц
target/q1/dense_2/BiasAddBiasAddtarget/q1/dense_2/MatMultarget/q1/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
|
target/q1/SqueezeSqueezetarget/q1/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
b
target/q1_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
Џ
target/q1_1/concatConcatV2Placeholder_2target/mul_1target/q1_1/concat/axis*
N*'
_output_shapes
:         h*

Tidx0*
T0
г
target/q1_1/dense/MatMulMatMultarget/q1_1/concattarget/q1/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
Б
target/q1_1/dense/BiasAddBiasAddtarget/q1_1/dense/MatMultarget/q1/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
l
target/q1_1/dense/ReluRelutarget/q1_1/dense/BiasAdd*(
_output_shapes
:         ђ*
T0
┤
target/q1_1/dense_1/MatMulMatMultarget/q1_1/dense/Relutarget/q1/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Е
target/q1_1/dense_1/BiasAddBiasAddtarget/q1_1/dense_1/MatMultarget/q1/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
p
target/q1_1/dense_1/ReluRelutarget/q1_1/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
х
target/q1_1/dense_2/MatMulMatMultarget/q1_1/dense_1/Relutarget/q1/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
е
target/q1_1/dense_2/BiasAddBiasAddtarget/q1_1/dense_2/MatMultarget/q1/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
ђ
target/q1_1/SqueezeSqueezetarget/q1_1/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
`
target/q2/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
ў
target/q2/concatConcatV2Placeholder_2Placeholder_1target/q2/concat/axis*
T0*
N*'
_output_shapes
:         h*

Tidx0
│
7target/q2/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"h      *)
_class
loc:@target/q2/dense/kernel*
dtype0*
_output_shapes
:
Ц
5target/q2/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *б]Ћй*)
_class
loc:@target/q2/dense/kernel*
dtype0*
_output_shapes
: 
Ц
5target/q2/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *б]Ћ=*)
_class
loc:@target/q2/dense/kernel*
dtype0*
_output_shapes
: 
Ё
?target/q2/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7target/q2/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@target/q2/dense/kernel*
seed2Ї*
dtype0*
_output_shapes
:	hђ
Ш
5target/q2/dense/kernel/Initializer/random_uniform/subSub5target/q2/dense/kernel/Initializer/random_uniform/max5target/q2/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/q2/dense/kernel*
_output_shapes
: 
Ѕ
5target/q2/dense/kernel/Initializer/random_uniform/mulMul?target/q2/dense/kernel/Initializer/random_uniform/RandomUniform5target/q2/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	hђ*
T0*)
_class
loc:@target/q2/dense/kernel
ч
1target/q2/dense/kernel/Initializer/random_uniformAdd5target/q2/dense/kernel/Initializer/random_uniform/mul5target/q2/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	hђ*
T0*)
_class
loc:@target/q2/dense/kernel
и
target/q2/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	hђ*
shared_name *)
_class
loc:@target/q2/dense/kernel*
	container *
shape:	hђ
­
target/q2/dense/kernel/AssignAssigntarget/q2/dense/kernel1target/q2/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
ћ
target/q2/dense/kernel/readIdentitytarget/q2/dense/kernel*
T0*)
_class
loc:@target/q2/dense/kernel*
_output_shapes
:	hђ
ф
6target/q2/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*'
_class
loc:@target/q2/dense/bias*
dtype0*
_output_shapes
:
џ
,target/q2/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@target/q2/dense/bias*
dtype0*
_output_shapes
: 
ш
&target/q2/dense/bias/Initializer/zerosFill6target/q2/dense/bias/Initializer/zeros/shape_as_tensor,target/q2/dense/bias/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@target/q2/dense/bias*
_output_shapes	
:ђ
Ф
target/q2/dense/bias
VariableV2*
shared_name *'
_class
loc:@target/q2/dense/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
█
target/q2/dense/bias/AssignAssigntarget/q2/dense/bias&target/q2/dense/bias/Initializer/zeros*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
і
target/q2/dense/bias/readIdentitytarget/q2/dense/bias*
_output_shapes	
:ђ*
T0*'
_class
loc:@target/q2/dense/bias
е
target/q2/dense/MatMulMatMultarget/q2/concattarget/q2/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ъ
target/q2/dense/BiasAddBiasAddtarget/q2/dense/MatMultarget/q2/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
h
target/q2/dense/ReluRelutarget/q2/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
и
9target/q2/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@target/q2/dense_1/kernel*
dtype0*
_output_shapes
:
Е
7target/q2/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *  ђй*+
_class!
loc:@target/q2/dense_1/kernel*
dtype0*
_output_shapes
: 
Е
7target/q2/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  ђ=*+
_class!
loc:@target/q2/dense_1/kernel*
dtype0*
_output_shapes
: 
ї
Atarget/q2/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q2/dense_1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђђ*

seed *
T0*+
_class!
loc:@target/q2/dense_1/kernel*
seed2а
■
7target/q2/dense_1/kernel/Initializer/random_uniform/subSub7target/q2/dense_1/kernel/Initializer/random_uniform/max7target/q2/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
_output_shapes
: 
њ
7target/q2/dense_1/kernel/Initializer/random_uniform/mulMulAtarget/q2/dense_1/kernel/Initializer/random_uniform/RandomUniform7target/q2/dense_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/q2/dense_1/kernel* 
_output_shapes
:
ђђ
ё
3target/q2/dense_1/kernel/Initializer/random_uniformAdd7target/q2/dense_1/kernel/Initializer/random_uniform/mul7target/q2/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
ђђ*
T0*+
_class!
loc:@target/q2/dense_1/kernel
й
target/q2/dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *+
_class!
loc:@target/q2/dense_1/kernel*
	container *
shape:
ђђ
щ
target/q2/dense_1/kernel/AssignAssigntarget/q2/dense_1/kernel3target/q2/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Џ
target/q2/dense_1/kernel/readIdentitytarget/q2/dense_1/kernel* 
_output_shapes
:
ђђ*
T0*+
_class!
loc:@target/q2/dense_1/kernel
б
(target/q2/dense_1/bias/Initializer/zerosConst*
valueBђ*    *)
_class
loc:@target/q2/dense_1/bias*
dtype0*
_output_shapes	
:ђ
»
target/q2/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *)
_class
loc:@target/q2/dense_1/bias*
	container *
shape:ђ
с
target/q2/dense_1/bias/AssignAssigntarget/q2/dense_1/bias(target/q2/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*)
_class
loc:@target/q2/dense_1/bias
љ
target/q2/dense_1/bias/readIdentitytarget/q2/dense_1/bias*
_output_shapes	
:ђ*
T0*)
_class
loc:@target/q2/dense_1/bias
░
target/q2/dense_1/MatMulMatMultarget/q2/dense/Relutarget/q2/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
Ц
target/q2/dense_1/BiasAddBiasAddtarget/q2/dense_1/MatMultarget/q2/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
l
target/q2/dense_1/ReluRelutarget/q2/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
и
9target/q2/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@target/q2/dense_2/kernel*
dtype0*
_output_shapes
:
Е
7target/q2/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *|Пй*+
_class!
loc:@target/q2/dense_2/kernel*
dtype0*
_output_shapes
: 
Е
7target/q2/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *|П=*+
_class!
loc:@target/q2/dense_2/kernel*
dtype0*
_output_shapes
: 
І
Atarget/q2/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q2/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0*+
_class!
loc:@target/q2/dense_2/kernel*
seed2▒
■
7target/q2/dense_2/kernel/Initializer/random_uniform/subSub7target/q2/dense_2/kernel/Initializer/random_uniform/max7target/q2/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
_output_shapes
: 
Љ
7target/q2/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/q2/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/q2/dense_2/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
_output_shapes
:	ђ
Ѓ
3target/q2/dense_2/kernel/Initializer/random_uniformAdd7target/q2/dense_2/kernel/Initializer/random_uniform/mul7target/q2/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
_output_shapes
:	ђ
╗
target/q2/dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	ђ*
shared_name *+
_class!
loc:@target/q2/dense_2/kernel*
	container *
shape:	ђ
Э
target/q2/dense_2/kernel/AssignAssigntarget/q2/dense_2/kernel3target/q2/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
џ
target/q2/dense_2/kernel/readIdentitytarget/q2/dense_2/kernel*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
_output_shapes
:	ђ
а
(target/q2/dense_2/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/q2/dense_2/bias*
dtype0*
_output_shapes
:
Г
target/q2/dense_2/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@target/q2/dense_2/bias*
	container 
Р
target/q2/dense_2/bias/AssignAssigntarget/q2/dense_2/bias(target/q2/dense_2/bias/Initializer/zeros*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ј
target/q2/dense_2/bias/readIdentitytarget/q2/dense_2/bias*
_output_shapes
:*
T0*)
_class
loc:@target/q2/dense_2/bias
▒
target/q2/dense_2/MatMulMatMultarget/q2/dense_1/Relutarget/q2/dense_2/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
ц
target/q2/dense_2/BiasAddBiasAddtarget/q2/dense_2/MatMultarget/q2/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
|
target/q2/SqueezeSqueezetarget/q2/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
b
target/q2_1/concat/axisConst*
valueB :
         *
dtype0*
_output_shapes
: 
Џ
target/q2_1/concatConcatV2Placeholder_2target/mul_1target/q2_1/concat/axis*
T0*
N*'
_output_shapes
:         h*

Tidx0
г
target/q2_1/dense/MatMulMatMultarget/q2_1/concattarget/q2/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
Б
target/q2_1/dense/BiasAddBiasAddtarget/q2_1/dense/MatMultarget/q2/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
l
target/q2_1/dense/ReluRelutarget/q2_1/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
┤
target/q2_1/dense_1/MatMulMatMultarget/q2_1/dense/Relutarget/q2/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Е
target/q2_1/dense_1/BiasAddBiasAddtarget/q2_1/dense_1/MatMultarget/q2/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
p
target/q2_1/dense_1/ReluRelutarget/q2_1/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
х
target/q2_1/dense_2/MatMulMatMultarget/q2_1/dense_1/Relutarget/q2/dense_2/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
е
target/q2_1/dense_2/BiasAddBiasAddtarget/q2_1/dense_2/MatMultarget/q2/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
ђ
target/q2_1/SqueezeSqueezetarget/q2_1/dense_2/BiasAdd*#
_output_shapes
:         *
squeeze_dims
*
T0
▒
6target/v/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"I      *(
_class
loc:@target/v/dense/kernel*
dtype0*
_output_shapes
:
Б
4target/v/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *%vЌй*(
_class
loc:@target/v/dense/kernel*
dtype0*
_output_shapes
: 
Б
4target/v/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *%vЌ=*(
_class
loc:@target/v/dense/kernel*
dtype0*
_output_shapes
: 
ѓ
>target/v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/v/dense/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@target/v/dense/kernel*
seed2═*
dtype0*
_output_shapes
:	Iђ*

seed 
Ы
4target/v/dense/kernel/Initializer/random_uniform/subSub4target/v/dense/kernel/Initializer/random_uniform/max4target/v/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*(
_class
loc:@target/v/dense/kernel
Ё
4target/v/dense/kernel/Initializer/random_uniform/mulMul>target/v/dense/kernel/Initializer/random_uniform/RandomUniform4target/v/dense/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/v/dense/kernel*
_output_shapes
:	Iђ
э
0target/v/dense/kernel/Initializer/random_uniformAdd4target/v/dense/kernel/Initializer/random_uniform/mul4target/v/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	Iђ*
T0*(
_class
loc:@target/v/dense/kernel
х
target/v/dense/kernel
VariableV2*
shared_name *(
_class
loc:@target/v/dense/kernel*
	container *
shape:	Iђ*
dtype0*
_output_shapes
:	Iђ
В
target/v/dense/kernel/AssignAssigntarget/v/dense/kernel0target/v/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@target/v/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
Љ
target/v/dense/kernel/readIdentitytarget/v/dense/kernel*
T0*(
_class
loc:@target/v/dense/kernel*
_output_shapes
:	Iђ
е
5target/v/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*&
_class
loc:@target/v/dense/bias*
dtype0*
_output_shapes
:
ў
+target/v/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *&
_class
loc:@target/v/dense/bias*
dtype0*
_output_shapes
: 
ы
%target/v/dense/bias/Initializer/zerosFill5target/v/dense/bias/Initializer/zeros/shape_as_tensor+target/v/dense/bias/Initializer/zeros/Const*
T0*

index_type0*&
_class
loc:@target/v/dense/bias*
_output_shapes	
:ђ
Е
target/v/dense/bias
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *&
_class
loc:@target/v/dense/bias*
	container 
О
target/v/dense/bias/AssignAssigntarget/v/dense/bias%target/v/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*&
_class
loc:@target/v/dense/bias
Є
target/v/dense/bias/readIdentitytarget/v/dense/bias*
T0*&
_class
loc:@target/v/dense/bias*
_output_shapes	
:ђ
Б
target/v/dense/MatMulMatMulPlaceholder_2target/v/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
ю
target/v/dense/BiasAddBiasAddtarget/v/dense/MatMultarget/v/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
f
target/v/dense/ReluRelutarget/v/dense/BiasAdd*
T0*(
_output_shapes
:         ђ
х
8target/v/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      **
_class 
loc:@target/v/dense_1/kernel*
dtype0*
_output_shapes
:
Д
6target/v/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *  ђй**
_class 
loc:@target/v/dense_1/kernel*
dtype0*
_output_shapes
: 
Д
6target/v/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  ђ=**
_class 
loc:@target/v/dense_1/kernel*
dtype0*
_output_shapes
: 
Ѕ
@target/v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform8target/v/dense_1/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@target/v/dense_1/kernel*
seed2Я*
dtype0* 
_output_shapes
:
ђђ*

seed 
Щ
6target/v/dense_1/kernel/Initializer/random_uniform/subSub6target/v/dense_1/kernel/Initializer/random_uniform/max6target/v/dense_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@target/v/dense_1/kernel*
_output_shapes
: 
ј
6target/v/dense_1/kernel/Initializer/random_uniform/mulMul@target/v/dense_1/kernel/Initializer/random_uniform/RandomUniform6target/v/dense_1/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@target/v/dense_1/kernel* 
_output_shapes
:
ђђ
ђ
2target/v/dense_1/kernel/Initializer/random_uniformAdd6target/v/dense_1/kernel/Initializer/random_uniform/mul6target/v/dense_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@target/v/dense_1/kernel* 
_output_shapes
:
ђђ
╗
target/v/dense_1/kernel
VariableV2*
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name **
_class 
loc:@target/v/dense_1/kernel*
	container 
ш
target/v/dense_1/kernel/AssignAssigntarget/v/dense_1/kernel2target/v/dense_1/kernel/Initializer/random_uniform*
T0**
_class 
loc:@target/v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
ў
target/v/dense_1/kernel/readIdentitytarget/v/dense_1/kernel*
T0**
_class 
loc:@target/v/dense_1/kernel* 
_output_shapes
:
ђђ
а
'target/v/dense_1/bias/Initializer/zerosConst*
valueBђ*    *(
_class
loc:@target/v/dense_1/bias*
dtype0*
_output_shapes	
:ђ
Г
target/v/dense_1/bias
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *(
_class
loc:@target/v/dense_1/bias*
	container 
▀
target/v/dense_1/bias/AssignAssigntarget/v/dense_1/bias'target/v/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*(
_class
loc:@target/v/dense_1/bias
Ї
target/v/dense_1/bias/readIdentitytarget/v/dense_1/bias*
T0*(
_class
loc:@target/v/dense_1/bias*
_output_shapes	
:ђ
Г
target/v/dense_1/MatMulMatMultarget/v/dense/Relutarget/v/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
б
target/v/dense_1/BiasAddBiasAddtarget/v/dense_1/MatMultarget/v/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
j
target/v/dense_1/ReluRelutarget/v/dense_1/BiasAdd*
T0*(
_output_shapes
:         ђ
х
8target/v/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      **
_class 
loc:@target/v/dense_2/kernel*
dtype0*
_output_shapes
:
Д
6target/v/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *|Пй**
_class 
loc:@target/v/dense_2/kernel
Д
6target/v/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *|П=**
_class 
loc:@target/v/dense_2/kernel*
dtype0*
_output_shapes
: 
ѕ
@target/v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform8target/v/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђ*

seed *
T0**
_class 
loc:@target/v/dense_2/kernel*
seed2ы
Щ
6target/v/dense_2/kernel/Initializer/random_uniform/subSub6target/v/dense_2/kernel/Initializer/random_uniform/max6target/v/dense_2/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@target/v/dense_2/kernel*
_output_shapes
: 
Ї
6target/v/dense_2/kernel/Initializer/random_uniform/mulMul@target/v/dense_2/kernel/Initializer/random_uniform/RandomUniform6target/v/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	ђ*
T0**
_class 
loc:@target/v/dense_2/kernel
 
2target/v/dense_2/kernel/Initializer/random_uniformAdd6target/v/dense_2/kernel/Initializer/random_uniform/mul6target/v/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	ђ*
T0**
_class 
loc:@target/v/dense_2/kernel
╣
target/v/dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	ђ*
shared_name **
_class 
loc:@target/v/dense_2/kernel*
	container *
shape:	ђ
З
target/v/dense_2/kernel/AssignAssigntarget/v/dense_2/kernel2target/v/dense_2/kernel/Initializer/random_uniform*
T0**
_class 
loc:@target/v/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
Ќ
target/v/dense_2/kernel/readIdentitytarget/v/dense_2/kernel*
_output_shapes
:	ђ*
T0**
_class 
loc:@target/v/dense_2/kernel
ъ
'target/v/dense_2/bias/Initializer/zerosConst*
valueB*    *(
_class
loc:@target/v/dense_2/bias*
dtype0*
_output_shapes
:
Ф
target/v/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *(
_class
loc:@target/v/dense_2/bias*
	container *
shape:
я
target/v/dense_2/bias/AssignAssigntarget/v/dense_2/bias'target/v/dense_2/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@target/v/dense_2/bias*
validate_shape(*
_output_shapes
:
ї
target/v/dense_2/bias/readIdentitytarget/v/dense_2/bias*
T0*(
_class
loc:@target/v/dense_2/bias*
_output_shapes
:
«
target/v/dense_2/MatMulMatMultarget/v/dense_1/Relutarget/v/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
А
target/v/dense_2/BiasAddBiasAddtarget/v/dense_2/MatMultarget/v/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
z
target/v/SqueezeSqueezetarget/v/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
f
MinimumMinimummain/q1_1/Squeezemain/q2_1/Squeeze*
T0*#
_output_shapes
:         
J
sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
N
subSubsub/xPlaceholder_4*
T0*#
_output_shapes
:         
J
mul/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
D
mulMulmul/xsub*
T0*#
_output_shapes
:         
Q
mul_1Mulmultarget/v/Squeeze*
T0*#
_output_shapes
:         
N
addAddPlaceholder_3mul_1*
T0*#
_output_shapes
:         
O
StopGradientStopGradientadd*
T0*#
_output_shapes
:         
]
mul_2MulPlaceholder_5main/pi/cond/Merge*
T0*#
_output_shapes
:         
J
sub_1SubMinimummul_2*
T0*#
_output_shapes
:         
S
StopGradient_1StopGradientsub_1*
T0*#
_output_shapes
:         
]
mul_3MulPlaceholder_5main/pi/cond/Merge*
T0*#
_output_shapes
:         
T
sub_2Submul_3main/q1_1/Squeeze*#
_output_shapes
:         *
T0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
X
MeanMeansub_2Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Y
sub_3SubStopGradientmain/q1/Squeeze*
T0*#
_output_shapes
:         
J
pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
F
powPowsub_3pow/y*
T0*#
_output_shapes
:         
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Z
Mean_1MeanpowConst_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
L
mul_4/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
>
mul_4Mulmul_4/xMean_1*
_output_shapes
: *
T0
Y
sub_4SubStopGradientmain/q2/Squeeze*
T0*#
_output_shapes
:         
L
pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
J
pow_1Powsub_4pow_1/y*#
_output_shapes
:         *
T0
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_2Meanpow_1Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
L
mul_5/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
>
mul_5Mulmul_5/xMean_2*
T0*
_output_shapes
: 
Z
sub_5SubStopGradient_1main/v/Squeeze*
T0*#
_output_shapes
:         
L
pow_2/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
J
pow_2Powsub_5pow_2/y*
T0*#
_output_shapes
:         
Q
Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_3Meanpow_2Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
L
mul_6/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
>
mul_6Mulmul_6/xMean_3*
T0*
_output_shapes
: 
;
add_1Addmul_4mul_5*
T0*
_output_shapes
: 
;
add_2Addadd_1mul_6*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ї
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
^
gradients/Mean_grad/ShapeShapesub_2*
T0*
out_type0*
_output_shapes
:
ў
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
`
gradients/Mean_grad/Shape_1Shapesub_2*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
ќ
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
џ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
ѓ
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
ђ
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
ѕ
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:         
_
gradients/sub_2_grad/ShapeShapemul_3*
T0*
out_type0*
_output_shapes
:
m
gradients/sub_2_grad/Shape_1Shapemain/q1_1/Squeeze*
T0*
out_type0*
_output_shapes
:
║
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
е
gradients/sub_2_grad/SumSumgradients/Mean_grad/truediv*gradients/sub_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ў
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
г
gradients/sub_2_grad/Sum_1Sumgradients/Mean_grad/truediv,gradients/sub_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
_output_shapes
:*
T0
Ю
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
я
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape*#
_output_shapes
:         
С
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*#
_output_shapes
:         
]
gradients/mul_3_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
n
gradients/mul_3_grad/Shape_1Shapemain/pi/cond/Merge*
T0*
out_type0*
_output_shapes
:
║
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
љ
gradients/mul_3_grad/MulMul-gradients/sub_2_grad/tuple/control_dependencymain/pi/cond/Merge*
T0*#
_output_shapes
:         
Ц
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ї
gradients/mul_3_grad/Mul_1MulPlaceholder_5-gradients/sub_2_grad/tuple/control_dependency*
T0*#
_output_shapes
:         
Ф
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ъ
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*#
_output_shapes
:         *
T0*
Tshape0
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
Л
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape
С
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*#
_output_shapes
:         

&gradients/main/q1_1/Squeeze_grad/ShapeShapemain/q1_1/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
╠
(gradients/main/q1_1/Squeeze_grad/ReshapeReshape/gradients/sub_2_grad/tuple/control_dependency_1&gradients/main/q1_1/Squeeze_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
В
+gradients/main/pi/cond/Merge_grad/cond_gradSwitch/gradients/mul_3_grad/tuple/control_dependency_1main/pi/cond/pred_id*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*2
_output_shapes 
:         :         
h
2gradients/main/pi/cond/Merge_grad/tuple/group_depsNoOp,^gradients/main/pi/cond/Merge_grad/cond_grad
Ѕ
:gradients/main/pi/cond/Merge_grad/tuple/control_dependencyIdentity+gradients/main/pi/cond/Merge_grad/cond_grad3^gradients/main/pi/cond/Merge_grad/tuple/group_deps*#
_output_shapes
:         *
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1
Ї
<gradients/main/pi/cond/Merge_grad/tuple/control_dependency_1Identity-gradients/main/pi/cond/Merge_grad/cond_grad:13^gradients/main/pi/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*#
_output_shapes
:         
Е
4gradients/main/q1_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients/main/q1_1/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
Б
9gradients/main/q1_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp)^gradients/main/q1_1/Squeeze_grad/Reshape5^gradients/main/q1_1/dense_2/BiasAdd_grad/BiasAddGrad
б
Agradients/main/q1_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients/main/q1_1/Squeeze_grad/Reshape:^gradients/main/q1_1/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*;
_class1
/-loc:@gradients/main/q1_1/Squeeze_grad/Reshape
»
Cgradients/main/q1_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/main/q1_1/dense_2/BiasAdd_grad/BiasAddGrad:^gradients/main/q1_1/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients/main/q1_1/dense_2/BiasAdd_grad/BiasAddGrad
ђ
'gradients/main/pi/cond/Sum_1_grad/ShapeShapemain/pi/cond/cond_1/Merge*
_output_shapes
:*
T0*
out_type0
ц
&gradients/main/pi/cond/Sum_1_grad/SizeConst*
value	B :*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
О
%gradients/main/pi/cond/Sum_1_grad/addAdd$main/pi/cond/Sum_1/reduction_indices&gradients/main/pi/cond/Sum_1_grad/Size*
T0*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
_output_shapes
: 
П
%gradients/main/pi/cond/Sum_1_grad/modFloorMod%gradients/main/pi/cond/Sum_1_grad/add&gradients/main/pi/cond/Sum_1_grad/Size*
T0*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
_output_shapes
: 
е
)gradients/main/pi/cond/Sum_1_grad/Shape_1Const*
valueB *:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
Ф
-gradients/main/pi/cond/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape
Ф
-gradients/main/pi/cond/Sum_1_grad/range/deltaConst*
value	B :*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
џ
'gradients/main/pi/cond/Sum_1_grad/rangeRange-gradients/main/pi/cond/Sum_1_grad/range/start&gradients/main/pi/cond/Sum_1_grad/Size-gradients/main/pi/cond/Sum_1_grad/range/delta*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
_output_shapes
:*

Tidx0
ф
,gradients/main/pi/cond/Sum_1_grad/Fill/valueConst*
value	B :*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
Ш
&gradients/main/pi/cond/Sum_1_grad/FillFill)gradients/main/pi/cond/Sum_1_grad/Shape_1,gradients/main/pi/cond/Sum_1_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape
╦
/gradients/main/pi/cond/Sum_1_grad/DynamicStitchDynamicStitch'gradients/main/pi/cond/Sum_1_grad/range%gradients/main/pi/cond/Sum_1_grad/mod'gradients/main/pi/cond/Sum_1_grad/Shape&gradients/main/pi/cond/Sum_1_grad/Fill*
T0*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
N*
_output_shapes
:
Е
+gradients/main/pi/cond/Sum_1_grad/Maximum/yConst*
value	B :*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
з
)gradients/main/pi/cond/Sum_1_grad/MaximumMaximum/gradients/main/pi/cond/Sum_1_grad/DynamicStitch+gradients/main/pi/cond/Sum_1_grad/Maximum/y*
_output_shapes
:*
T0*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape
в
*gradients/main/pi/cond/Sum_1_grad/floordivFloorDiv'gradients/main/pi/cond/Sum_1_grad/Shape)gradients/main/pi/cond/Sum_1_grad/Maximum*
T0*:
_class0
.,loc:@gradients/main/pi/cond/Sum_1_grad/Shape*
_output_shapes
:
Ж
)gradients/main/pi/cond/Sum_1_grad/ReshapeReshape:gradients/main/pi/cond/Merge_grad/tuple/control_dependency/gradients/main/pi/cond/Sum_1_grad/DynamicStitch*0
_output_shapes
:                  *
T0*
Tshape0
╔
&gradients/main/pi/cond/Sum_1_grad/TileTile)gradients/main/pi/cond/Sum_1_grad/Reshape*gradients/main/pi/cond/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:         
|
%gradients/main/pi/cond/Sum_grad/ShapeShapemain/pi/cond/cond/Merge*
T0*
out_type0*
_output_shapes
:
а
$gradients/main/pi/cond/Sum_grad/SizeConst*
value	B :*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
dtype0*
_output_shapes
: 
¤
#gradients/main/pi/cond/Sum_grad/addAdd"main/pi/cond/Sum/reduction_indices$gradients/main/pi/cond/Sum_grad/Size*
T0*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
_output_shapes
: 
Н
#gradients/main/pi/cond/Sum_grad/modFloorMod#gradients/main/pi/cond/Sum_grad/add$gradients/main/pi/cond/Sum_grad/Size*
T0*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
_output_shapes
: 
ц
'gradients/main/pi/cond/Sum_grad/Shape_1Const*
valueB *8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Д
+gradients/main/pi/cond/Sum_grad/range/startConst*
value	B : *8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Д
+gradients/main/pi/cond/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape
љ
%gradients/main/pi/cond/Sum_grad/rangeRange+gradients/main/pi/cond/Sum_grad/range/start$gradients/main/pi/cond/Sum_grad/Size+gradients/main/pi/cond/Sum_grad/range/delta*

Tidx0*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
_output_shapes
:
д
*gradients/main/pi/cond/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape
Ь
$gradients/main/pi/cond/Sum_grad/FillFill'gradients/main/pi/cond/Sum_grad/Shape_1*gradients/main/pi/cond/Sum_grad/Fill/value*
T0*

index_type0*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
_output_shapes
: 
┐
-gradients/main/pi/cond/Sum_grad/DynamicStitchDynamicStitch%gradients/main/pi/cond/Sum_grad/range#gradients/main/pi/cond/Sum_grad/mod%gradients/main/pi/cond/Sum_grad/Shape$gradients/main/pi/cond/Sum_grad/Fill*
T0*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
N*
_output_shapes
:
Ц
)gradients/main/pi/cond/Sum_grad/Maximum/yConst*
value	B :*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
dtype0*
_output_shapes
: 
в
'gradients/main/pi/cond/Sum_grad/MaximumMaximum-gradients/main/pi/cond/Sum_grad/DynamicStitch)gradients/main/pi/cond/Sum_grad/Maximum/y*
T0*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
_output_shapes
:
с
(gradients/main/pi/cond/Sum_grad/floordivFloorDiv%gradients/main/pi/cond/Sum_grad/Shape'gradients/main/pi/cond/Sum_grad/Maximum*
T0*8
_class.
,*loc:@gradients/main/pi/cond/Sum_grad/Shape*
_output_shapes
:
У
'gradients/main/pi/cond/Sum_grad/ReshapeReshape<gradients/main/pi/cond/Merge_grad/tuple/control_dependency_1-gradients/main/pi/cond/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:                  
├
$gradients/main/pi/cond/Sum_grad/TileTile'gradients/main/pi/cond/Sum_grad/Reshape(gradients/main/pi/cond/Sum_grad/floordiv*
T0*'
_output_shapes
:         *

Tmultiples0
ы
.gradients/main/q1_1/dense_2/MatMul_grad/MatMulMatMulAgradients/main/q1_1/dense_2/BiasAdd_grad/tuple/control_dependencymain/q1/dense_2/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
т
0gradients/main/q1_1/dense_2/MatMul_grad/MatMul_1MatMulmain/q1_1/dense_1/ReluAgradients/main/q1_1/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( *
T0
ц
8gradients/main/q1_1/dense_2/MatMul_grad/tuple/group_depsNoOp/^gradients/main/q1_1/dense_2/MatMul_grad/MatMul1^gradients/main/q1_1/dense_2/MatMul_grad/MatMul_1
Г
@gradients/main/q1_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients/main/q1_1/dense_2/MatMul_grad/MatMul9^gradients/main/q1_1/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/q1_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         ђ
ф
Bgradients/main/q1_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients/main/q1_1/dense_2/MatMul_grad/MatMul_19^gradients/main/q1_1/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	ђ*
T0*C
_class9
75loc:@gradients/main/q1_1/dense_2/MatMul_grad/MatMul_1
Ђ
2gradients/main/pi/cond/cond_1/Merge_grad/cond_gradSwitch&gradients/main/pi/cond/Sum_1_grad/Tilemain/pi/cond/cond_1/pred_id*
T0*9
_class/
-+loc:@gradients/main/pi/cond/Sum_1_grad/Tile*:
_output_shapes(
&:         :         
v
9gradients/main/pi/cond/cond_1/Merge_grad/tuple/group_depsNoOp3^gradients/main/pi/cond/cond_1/Merge_grad/cond_grad
ф
Agradients/main/pi/cond/cond_1/Merge_grad/tuple/control_dependencyIdentity2gradients/main/pi/cond/cond_1/Merge_grad/cond_grad:^gradients/main/pi/cond/cond_1/Merge_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/cond/Sum_1_grad/Tile*'
_output_shapes
:         
«
Cgradients/main/pi/cond/cond_1/Merge_grad/tuple/control_dependency_1Identity4gradients/main/pi/cond/cond_1/Merge_grad/cond_grad:1:^gradients/main/pi/cond/cond_1/Merge_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/cond/Sum_1_grad/Tile*'
_output_shapes
:         
щ
0gradients/main/pi/cond/cond/Merge_grad/cond_gradSwitch$gradients/main/pi/cond/Sum_grad/Tilemain/pi/cond/cond/pred_id*
T0*7
_class-
+)loc:@gradients/main/pi/cond/Sum_grad/Tile*:
_output_shapes(
&:         :         
r
7gradients/main/pi/cond/cond/Merge_grad/tuple/group_depsNoOp1^gradients/main/pi/cond/cond/Merge_grad/cond_grad
б
?gradients/main/pi/cond/cond/Merge_grad/tuple/control_dependencyIdentity0gradients/main/pi/cond/cond/Merge_grad/cond_grad8^gradients/main/pi/cond/cond/Merge_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/cond/Sum_grad/Tile*'
_output_shapes
:         
д
Agradients/main/pi/cond/cond/Merge_grad/tuple/control_dependency_1Identity2gradients/main/pi/cond/cond/Merge_grad/cond_grad:18^gradients/main/pi/cond/cond/Merge_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/cond/Sum_grad/Tile*'
_output_shapes
:         
К
.gradients/main/q1_1/dense_1/Relu_grad/ReluGradReluGrad@gradients/main/q1_1/dense_2/MatMul_grad/tuple/control_dependencymain/q1_1/dense_1/Relu*
T0*(
_output_shapes
:         ђ
Ѕ
0gradients/main/pi/cond/cond_1/truediv_grad/ShapeShapemain/pi/cond/cond_1/sub_1*
T0*
out_type0*
_output_shapes
:
u
2gradients/main/pi/cond/cond_1/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ч
@gradients/main/pi/cond/cond_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients/main/pi/cond/cond_1/truediv_grad/Shape2gradients/main/pi/cond/cond_1/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
═
2gradients/main/pi/cond/cond_1/truediv_grad/RealDivRealDivAgradients/main/pi/cond/cond_1/Merge_grad/tuple/control_dependencymain/pi/cond/cond_1/sub_2*
T0*'
_output_shapes
:         
в
.gradients/main/pi/cond/cond_1/truediv_grad/SumSum2gradients/main/pi/cond/cond_1/truediv_grad/RealDiv@gradients/main/pi/cond/cond_1/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
▀
2gradients/main/pi/cond/cond_1/truediv_grad/ReshapeReshape.gradients/main/pi/cond/cond_1/truediv_grad/Sum0gradients/main/pi/cond/cond_1/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ѓ
.gradients/main/pi/cond/cond_1/truediv_grad/NegNegmain/pi/cond/cond_1/sub_1*
T0*'
_output_shapes
:         
╝
4gradients/main/pi/cond/cond_1/truediv_grad/RealDiv_1RealDiv.gradients/main/pi/cond/cond_1/truediv_grad/Negmain/pi/cond/cond_1/sub_2*
T0*'
_output_shapes
:         
┬
4gradients/main/pi/cond/cond_1/truediv_grad/RealDiv_2RealDiv4gradients/main/pi/cond/cond_1/truediv_grad/RealDiv_1main/pi/cond/cond_1/sub_2*'
_output_shapes
:         *
T0
Я
.gradients/main/pi/cond/cond_1/truediv_grad/mulMulAgradients/main/pi/cond/cond_1/Merge_grad/tuple/control_dependency4gradients/main/pi/cond/cond_1/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:         
в
0gradients/main/pi/cond/cond_1/truediv_grad/Sum_1Sum.gradients/main/pi/cond/cond_1/truediv_grad/mulBgradients/main/pi/cond/cond_1/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
н
4gradients/main/pi/cond/cond_1/truediv_grad/Reshape_1Reshape0gradients/main/pi/cond/cond_1/truediv_grad/Sum_12gradients/main/pi/cond/cond_1/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
»
;gradients/main/pi/cond/cond_1/truediv_grad/tuple/group_depsNoOp3^gradients/main/pi/cond/cond_1/truediv_grad/Reshape5^gradients/main/pi/cond/cond_1/truediv_grad/Reshape_1
║
Cgradients/main/pi/cond/cond_1/truediv_grad/tuple/control_dependencyIdentity2gradients/main/pi/cond/cond_1/truediv_grad/Reshape<^gradients/main/pi/cond/cond_1/truediv_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/main/pi/cond/cond_1/truediv_grad/Reshape*'
_output_shapes
:         
»
Egradients/main/pi/cond/cond_1/truediv_grad/tuple/control_dependency_1Identity4gradients/main/pi/cond/cond_1/truediv_grad/Reshape_1<^gradients/main/pi/cond/cond_1/truediv_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/main/pi/cond/cond_1/truediv_grad/Reshape_1*
_output_shapes
: 
┘
1gradients/main/pi/cond/cond_1/Log_grad/Reciprocal
Reciprocal main/pi/cond/cond_1/Log/Switch:1D^gradients/main/pi/cond/cond_1/Merge_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
█
*gradients/main/pi/cond/cond_1/Log_grad/mulMulCgradients/main/pi/cond/cond_1/Merge_grad/tuple/control_dependency_11gradients/main/pi/cond/cond_1/Log_grad/Reciprocal*
T0*'
_output_shapes
:         
Ё
.gradients/main/pi/cond/cond/truediv_grad/ShapeShapemain/pi/cond/cond/sub_1*
T0*
out_type0*
_output_shapes
:
s
0gradients/main/pi/cond/cond/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ш
>gradients/main/pi/cond/cond/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/main/pi/cond/cond/truediv_grad/Shape0gradients/main/pi/cond/cond/truediv_grad/Shape_1*
T0*2
_output_shapes 
:         :         
К
0gradients/main/pi/cond/cond/truediv_grad/RealDivRealDiv?gradients/main/pi/cond/cond/Merge_grad/tuple/control_dependencymain/pi/cond/cond/sub_2*'
_output_shapes
:         *
T0
т
,gradients/main/pi/cond/cond/truediv_grad/SumSum0gradients/main/pi/cond/cond/truediv_grad/RealDiv>gradients/main/pi/cond/cond/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┘
0gradients/main/pi/cond/cond/truediv_grad/ReshapeReshape,gradients/main/pi/cond/cond/truediv_grad/Sum.gradients/main/pi/cond/cond/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
~
,gradients/main/pi/cond/cond/truediv_grad/NegNegmain/pi/cond/cond/sub_1*
T0*'
_output_shapes
:         
Х
2gradients/main/pi/cond/cond/truediv_grad/RealDiv_1RealDiv,gradients/main/pi/cond/cond/truediv_grad/Negmain/pi/cond/cond/sub_2*
T0*'
_output_shapes
:         
╝
2gradients/main/pi/cond/cond/truediv_grad/RealDiv_2RealDiv2gradients/main/pi/cond/cond/truediv_grad/RealDiv_1main/pi/cond/cond/sub_2*'
_output_shapes
:         *
T0
┌
,gradients/main/pi/cond/cond/truediv_grad/mulMul?gradients/main/pi/cond/cond/Merge_grad/tuple/control_dependency2gradients/main/pi/cond/cond/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:         
т
.gradients/main/pi/cond/cond/truediv_grad/Sum_1Sum,gradients/main/pi/cond/cond/truediv_grad/mul@gradients/main/pi/cond/cond/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╬
2gradients/main/pi/cond/cond/truediv_grad/Reshape_1Reshape.gradients/main/pi/cond/cond/truediv_grad/Sum_10gradients/main/pi/cond/cond/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Е
9gradients/main/pi/cond/cond/truediv_grad/tuple/group_depsNoOp1^gradients/main/pi/cond/cond/truediv_grad/Reshape3^gradients/main/pi/cond/cond/truediv_grad/Reshape_1
▓
Agradients/main/pi/cond/cond/truediv_grad/tuple/control_dependencyIdentity0gradients/main/pi/cond/cond/truediv_grad/Reshape:^gradients/main/pi/cond/cond/truediv_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/main/pi/cond/cond/truediv_grad/Reshape*'
_output_shapes
:         
Д
Cgradients/main/pi/cond/cond/truediv_grad/tuple/control_dependency_1Identity2gradients/main/pi/cond/cond/truediv_grad/Reshape_1:^gradients/main/pi/cond/cond/truediv_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/main/pi/cond/cond/truediv_grad/Reshape_1*
_output_shapes
: 
М
/gradients/main/pi/cond/cond/Log_grad/Reciprocal
Reciprocalmain/pi/cond/cond/Log/Switch:1B^gradients/main/pi/cond/cond/Merge_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
Н
(gradients/main/pi/cond/cond/Log_grad/mulMulAgradients/main/pi/cond/cond/Merge_grad/tuple/control_dependency_1/gradients/main/pi/cond/cond/Log_grad/Reciprocal*
T0*'
_output_shapes
:         
░
4gradients/main/q1_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/main/q1_1/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:ђ*
T0
Е
9gradients/main/q1_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp5^gradients/main/q1_1/dense_1/BiasAdd_grad/BiasAddGrad/^gradients/main/q1_1/dense_1/Relu_grad/ReluGrad
»
Agradients/main/q1_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/main/q1_1/dense_1/Relu_grad/ReluGrad:^gradients/main/q1_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/q1_1/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:         ђ
░
Cgradients/main/q1_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/main/q1_1/dense_1/BiasAdd_grad/BiasAddGrad:^gradients/main/q1_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/main/q1_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Ё
.gradients/main/pi/cond/cond_1/sub_1_grad/ShapeShapemain/pi/cond/cond_1/Pow*
_output_shapes
:*
T0*
out_type0
s
0gradients/main/pi/cond/cond_1/sub_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Ш
>gradients/main/pi/cond/cond_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/main/pi/cond/cond_1/sub_1_grad/Shape0gradients/main/pi/cond/cond_1/sub_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Э
,gradients/main/pi/cond/cond_1/sub_1_grad/SumSumCgradients/main/pi/cond/cond_1/truediv_grad/tuple/control_dependency>gradients/main/pi/cond/cond_1/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┘
0gradients/main/pi/cond/cond_1/sub_1_grad/ReshapeReshape,gradients/main/pi/cond/cond_1/sub_1_grad/Sum.gradients/main/pi/cond/cond_1/sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ч
.gradients/main/pi/cond/cond_1/sub_1_grad/Sum_1SumCgradients/main/pi/cond/cond_1/truediv_grad/tuple/control_dependency@gradients/main/pi/cond/cond_1/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
,gradients/main/pi/cond/cond_1/sub_1_grad/NegNeg.gradients/main/pi/cond/cond_1/sub_1_grad/Sum_1*
T0*
_output_shapes
:
╠
2gradients/main/pi/cond/cond_1/sub_1_grad/Reshape_1Reshape,gradients/main/pi/cond/cond_1/sub_1_grad/Neg0gradients/main/pi/cond/cond_1/sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Е
9gradients/main/pi/cond/cond_1/sub_1_grad/tuple/group_depsNoOp1^gradients/main/pi/cond/cond_1/sub_1_grad/Reshape3^gradients/main/pi/cond/cond_1/sub_1_grad/Reshape_1
▓
Agradients/main/pi/cond/cond_1/sub_1_grad/tuple/control_dependencyIdentity0gradients/main/pi/cond/cond_1/sub_1_grad/Reshape:^gradients/main/pi/cond/cond_1/sub_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*C
_class9
75loc:@gradients/main/pi/cond/cond_1/sub_1_grad/Reshape
Д
Cgradients/main/pi/cond/cond_1/sub_1_grad/tuple/control_dependency_1Identity2gradients/main/pi/cond/cond_1/sub_1_grad/Reshape_1:^gradients/main/pi/cond/cond_1/sub_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/main/pi/cond/cond_1/sub_1_grad/Reshape_1*
_output_shapes
: 
ћ
gradients/SwitchSwitchmain/pi/cond/Maximum_1main/pi/cond/cond_1/pred_id*:
_output_shapes(
&:         :         *
T0
b
gradients/IdentityIdentitygradients/Switch*
T0*'
_output_shapes
:         
a
gradients/Shape_1Shapegradients/Switch*
T0*
out_type0*
_output_shapes
:
o
gradients/zeros/ConstConst^gradients/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
Ё
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
║
7gradients/main/pi/cond/cond_1/Log/Switch_grad/cond_gradMergegradients/zeros*gradients/main/pi/cond/cond_1/Log_grad/mul*
T0*
N*)
_output_shapes
:         : 
Ђ
,gradients/main/pi/cond/cond/sub_1_grad/ShapeShapemain/pi/cond/cond/Pow*
T0*
out_type0*
_output_shapes
:
q
.gradients/main/pi/cond/cond/sub_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
­
<gradients/main/pi/cond/cond/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/main/pi/cond/cond/sub_1_grad/Shape.gradients/main/pi/cond/cond/sub_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ы
*gradients/main/pi/cond/cond/sub_1_grad/SumSumAgradients/main/pi/cond/cond/truediv_grad/tuple/control_dependency<gradients/main/pi/cond/cond/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
М
.gradients/main/pi/cond/cond/sub_1_grad/ReshapeReshape*gradients/main/pi/cond/cond/sub_1_grad/Sum,gradients/main/pi/cond/cond/sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ш
,gradients/main/pi/cond/cond/sub_1_grad/Sum_1SumAgradients/main/pi/cond/cond/truediv_grad/tuple/control_dependency>gradients/main/pi/cond/cond/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ѓ
*gradients/main/pi/cond/cond/sub_1_grad/NegNeg,gradients/main/pi/cond/cond/sub_1_grad/Sum_1*
T0*
_output_shapes
:
к
0gradients/main/pi/cond/cond/sub_1_grad/Reshape_1Reshape*gradients/main/pi/cond/cond/sub_1_grad/Neg.gradients/main/pi/cond/cond/sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Б
7gradients/main/pi/cond/cond/sub_1_grad/tuple/group_depsNoOp/^gradients/main/pi/cond/cond/sub_1_grad/Reshape1^gradients/main/pi/cond/cond/sub_1_grad/Reshape_1
ф
?gradients/main/pi/cond/cond/sub_1_grad/tuple/control_dependencyIdentity.gradients/main/pi/cond/cond/sub_1_grad/Reshape8^gradients/main/pi/cond/cond/sub_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/pi/cond/cond/sub_1_grad/Reshape*'
_output_shapes
:         
Ъ
Agradients/main/pi/cond/cond/sub_1_grad/tuple/control_dependency_1Identity0gradients/main/pi/cond/cond/sub_1_grad/Reshape_18^gradients/main/pi/cond/cond/sub_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/main/pi/cond/cond/sub_1_grad/Reshape_1*
_output_shapes
: 
њ
gradients/Switch_1Switchmain/pi/cond/Maximummain/pi/cond/cond/pred_id*:
_output_shapes(
&:         :         *
T0
f
gradients/Identity_1Identitygradients/Switch_1*
T0*'
_output_shapes
:         
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
out_type0*
_output_shapes
:
s
gradients/zeros_1/ConstConst^gradients/Identity_1*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѕ
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*

index_type0*'
_output_shapes
:         
И
5gradients/main/pi/cond/cond/Log/Switch_grad/cond_gradMergegradients/zeros_1(gradients/main/pi/cond/cond/Log_grad/mul*
T0*
N*)
_output_shapes
:         : 
ы
.gradients/main/q1_1/dense_1/MatMul_grad/MatMulMatMulAgradients/main/q1_1/dense_1/BiasAdd_grad/tuple/control_dependencymain/q1/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
С
0gradients/main/q1_1/dense_1/MatMul_grad/MatMul_1MatMulmain/q1_1/dense/ReluAgradients/main/q1_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
ђђ*
transpose_a(
ц
8gradients/main/q1_1/dense_1/MatMul_grad/tuple/group_depsNoOp/^gradients/main/q1_1/dense_1/MatMul_grad/MatMul1^gradients/main/q1_1/dense_1/MatMul_grad/MatMul_1
Г
@gradients/main/q1_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients/main/q1_1/dense_1/MatMul_grad/MatMul9^gradients/main/q1_1/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/q1_1/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Ф
Bgradients/main/q1_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients/main/q1_1/dense_1/MatMul_grad/MatMul_19^gradients/main/q1_1/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
ђђ*
T0*C
_class9
75loc:@gradients/main/q1_1/dense_1/MatMul_grad/MatMul_1
і
,gradients/main/pi/cond/cond_1/Pow_grad/ShapeShapemain/pi/cond/cond_1/Pow/Switch*
T0*
out_type0*
_output_shapes
:
q
.gradients/main/pi/cond/cond_1/Pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
­
<gradients/main/pi/cond/cond_1/Pow_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/main/pi/cond/cond_1/Pow_grad/Shape.gradients/main/pi/cond/cond_1/Pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┐
*gradients/main/pi/cond/cond_1/Pow_grad/mulMulAgradients/main/pi/cond/cond_1/sub_1_grad/tuple/control_dependencymain/pi/cond/cond_1/sub*
T0*'
_output_shapes
:         
q
,gradients/main/pi/cond/cond_1/Pow_grad/sub/yConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ў
*gradients/main/pi/cond/cond_1/Pow_grad/subSubmain/pi/cond/cond_1/sub,gradients/main/pi/cond/cond_1/Pow_grad/sub/y*
T0*
_output_shapes
: 
»
*gradients/main/pi/cond/cond_1/Pow_grad/PowPowmain/pi/cond/cond_1/Pow/Switch*gradients/main/pi/cond/cond_1/Pow_grad/sub*'
_output_shapes
:         *
T0
й
,gradients/main/pi/cond/cond_1/Pow_grad/mul_1Mul*gradients/main/pi/cond/cond_1/Pow_grad/mul*gradients/main/pi/cond/cond_1/Pow_grad/Pow*'
_output_shapes
:         *
T0
П
*gradients/main/pi/cond/cond_1/Pow_grad/SumSum,gradients/main/pi/cond/cond_1/Pow_grad/mul_1<gradients/main/pi/cond/cond_1/Pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
М
.gradients/main/pi/cond/cond_1/Pow_grad/ReshapeReshape*gradients/main/pi/cond/cond_1/Pow_grad/Sum,gradients/main/pi/cond/cond_1/Pow_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
u
0gradients/main/pi/cond/cond_1/Pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
й
.gradients/main/pi/cond/cond_1/Pow_grad/GreaterGreatermain/pi/cond/cond_1/Pow/Switch0gradients/main/pi/cond/cond_1/Pow_grad/Greater/y*
T0*'
_output_shapes
:         
ћ
6gradients/main/pi/cond/cond_1/Pow_grad/ones_like/ShapeShapemain/pi/cond/cond_1/Pow/Switch*
T0*
out_type0*
_output_shapes
:
{
6gradients/main/pi/cond/cond_1/Pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
В
0gradients/main/pi/cond/cond_1/Pow_grad/ones_likeFill6gradients/main/pi/cond/cond_1/Pow_grad/ones_like/Shape6gradients/main/pi/cond/cond_1/Pow_grad/ones_like/Const*'
_output_shapes
:         *
T0*

index_type0
в
-gradients/main/pi/cond/cond_1/Pow_grad/SelectSelect.gradients/main/pi/cond/cond_1/Pow_grad/Greatermain/pi/cond/cond_1/Pow/Switch0gradients/main/pi/cond/cond_1/Pow_grad/ones_like*
T0*'
_output_shapes
:         
њ
*gradients/main/pi/cond/cond_1/Pow_grad/LogLog-gradients/main/pi/cond/cond_1/Pow_grad/Select*
T0*'
_output_shapes
:         
љ
1gradients/main/pi/cond/cond_1/Pow_grad/zeros_like	ZerosLikemain/pi/cond/cond_1/Pow/Switch*
T0*'
_output_shapes
:         
Щ
/gradients/main/pi/cond/cond_1/Pow_grad/Select_1Select.gradients/main/pi/cond/cond_1/Pow_grad/Greater*gradients/main/pi/cond/cond_1/Pow_grad/Log1gradients/main/pi/cond/cond_1/Pow_grad/zeros_like*'
_output_shapes
:         *
T0
┴
,gradients/main/pi/cond/cond_1/Pow_grad/mul_2MulAgradients/main/pi/cond/cond_1/sub_1_grad/tuple/control_dependencymain/pi/cond/cond_1/Pow*
T0*'
_output_shapes
:         
─
,gradients/main/pi/cond/cond_1/Pow_grad/mul_3Mul,gradients/main/pi/cond/cond_1/Pow_grad/mul_2/gradients/main/pi/cond/cond_1/Pow_grad/Select_1*'
_output_shapes
:         *
T0
р
,gradients/main/pi/cond/cond_1/Pow_grad/Sum_1Sum,gradients/main/pi/cond/cond_1/Pow_grad/mul_3>gradients/main/pi/cond/cond_1/Pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╚
0gradients/main/pi/cond/cond_1/Pow_grad/Reshape_1Reshape,gradients/main/pi/cond/cond_1/Pow_grad/Sum_1.gradients/main/pi/cond/cond_1/Pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Б
7gradients/main/pi/cond/cond_1/Pow_grad/tuple/group_depsNoOp/^gradients/main/pi/cond/cond_1/Pow_grad/Reshape1^gradients/main/pi/cond/cond_1/Pow_grad/Reshape_1
ф
?gradients/main/pi/cond/cond_1/Pow_grad/tuple/control_dependencyIdentity.gradients/main/pi/cond/cond_1/Pow_grad/Reshape8^gradients/main/pi/cond/cond_1/Pow_grad/tuple/group_deps*'
_output_shapes
:         *
T0*A
_class7
53loc:@gradients/main/pi/cond/cond_1/Pow_grad/Reshape
Ъ
Agradients/main/pi/cond/cond_1/Pow_grad/tuple/control_dependency_1Identity0gradients/main/pi/cond/cond_1/Pow_grad/Reshape_18^gradients/main/pi/cond/cond_1/Pow_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/main/pi/cond/cond_1/Pow_grad/Reshape_1*
_output_shapes
: 
є
*gradients/main/pi/cond/cond/Pow_grad/ShapeShapemain/pi/cond/cond/Pow/Switch*
T0*
out_type0*
_output_shapes
:
o
,gradients/main/pi/cond/cond/Pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ж
:gradients/main/pi/cond/cond/Pow_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/main/pi/cond/cond/Pow_grad/Shape,gradients/main/pi/cond/cond/Pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╣
(gradients/main/pi/cond/cond/Pow_grad/mulMul?gradients/main/pi/cond/cond/sub_1_grad/tuple/control_dependencymain/pi/cond/cond/sub*'
_output_shapes
:         *
T0
o
*gradients/main/pi/cond/cond/Pow_grad/sub/yConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Њ
(gradients/main/pi/cond/cond/Pow_grad/subSubmain/pi/cond/cond/sub*gradients/main/pi/cond/cond/Pow_grad/sub/y*
T0*
_output_shapes
: 
Е
(gradients/main/pi/cond/cond/Pow_grad/PowPowmain/pi/cond/cond/Pow/Switch(gradients/main/pi/cond/cond/Pow_grad/sub*
T0*'
_output_shapes
:         
и
*gradients/main/pi/cond/cond/Pow_grad/mul_1Mul(gradients/main/pi/cond/cond/Pow_grad/mul(gradients/main/pi/cond/cond/Pow_grad/Pow*
T0*'
_output_shapes
:         
О
(gradients/main/pi/cond/cond/Pow_grad/SumSum*gradients/main/pi/cond/cond/Pow_grad/mul_1:gradients/main/pi/cond/cond/Pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
═
,gradients/main/pi/cond/cond/Pow_grad/ReshapeReshape(gradients/main/pi/cond/cond/Pow_grad/Sum*gradients/main/pi/cond/cond/Pow_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
s
.gradients/main/pi/cond/cond/Pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
и
,gradients/main/pi/cond/cond/Pow_grad/GreaterGreatermain/pi/cond/cond/Pow/Switch.gradients/main/pi/cond/cond/Pow_grad/Greater/y*'
_output_shapes
:         *
T0
љ
4gradients/main/pi/cond/cond/Pow_grad/ones_like/ShapeShapemain/pi/cond/cond/Pow/Switch*
T0*
out_type0*
_output_shapes
:
y
4gradients/main/pi/cond/cond/Pow_grad/ones_like/ConstConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Т
.gradients/main/pi/cond/cond/Pow_grad/ones_likeFill4gradients/main/pi/cond/cond/Pow_grad/ones_like/Shape4gradients/main/pi/cond/cond/Pow_grad/ones_like/Const*'
_output_shapes
:         *
T0*

index_type0
с
+gradients/main/pi/cond/cond/Pow_grad/SelectSelect,gradients/main/pi/cond/cond/Pow_grad/Greatermain/pi/cond/cond/Pow/Switch.gradients/main/pi/cond/cond/Pow_grad/ones_like*'
_output_shapes
:         *
T0
ј
(gradients/main/pi/cond/cond/Pow_grad/LogLog+gradients/main/pi/cond/cond/Pow_grad/Select*
T0*'
_output_shapes
:         
ї
/gradients/main/pi/cond/cond/Pow_grad/zeros_like	ZerosLikemain/pi/cond/cond/Pow/Switch*
T0*'
_output_shapes
:         
Ы
-gradients/main/pi/cond/cond/Pow_grad/Select_1Select,gradients/main/pi/cond/cond/Pow_grad/Greater(gradients/main/pi/cond/cond/Pow_grad/Log/gradients/main/pi/cond/cond/Pow_grad/zeros_like*'
_output_shapes
:         *
T0
╗
*gradients/main/pi/cond/cond/Pow_grad/mul_2Mul?gradients/main/pi/cond/cond/sub_1_grad/tuple/control_dependencymain/pi/cond/cond/Pow*
T0*'
_output_shapes
:         
Й
*gradients/main/pi/cond/cond/Pow_grad/mul_3Mul*gradients/main/pi/cond/cond/Pow_grad/mul_2-gradients/main/pi/cond/cond/Pow_grad/Select_1*'
_output_shapes
:         *
T0
█
*gradients/main/pi/cond/cond/Pow_grad/Sum_1Sum*gradients/main/pi/cond/cond/Pow_grad/mul_3<gradients/main/pi/cond/cond/Pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┬
.gradients/main/pi/cond/cond/Pow_grad/Reshape_1Reshape*gradients/main/pi/cond/cond/Pow_grad/Sum_1,gradients/main/pi/cond/cond/Pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ю
5gradients/main/pi/cond/cond/Pow_grad/tuple/group_depsNoOp-^gradients/main/pi/cond/cond/Pow_grad/Reshape/^gradients/main/pi/cond/cond/Pow_grad/Reshape_1
б
=gradients/main/pi/cond/cond/Pow_grad/tuple/control_dependencyIdentity,gradients/main/pi/cond/cond/Pow_grad/Reshape6^gradients/main/pi/cond/cond/Pow_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/cond/cond/Pow_grad/Reshape*'
_output_shapes
:         
Ќ
?gradients/main/pi/cond/cond/Pow_grad/tuple/control_dependency_1Identity.gradients/main/pi/cond/cond/Pow_grad/Reshape_16^gradients/main/pi/cond/cond/Pow_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/pi/cond/cond/Pow_grad/Reshape_1*
_output_shapes
: 
├
,gradients/main/q1_1/dense/Relu_grad/ReluGradReluGrad@gradients/main/q1_1/dense_1/MatMul_grad/tuple/control_dependencymain/q1_1/dense/Relu*(
_output_shapes
:         ђ*
T0
ќ
gradients/Switch_2Switchmain/pi/cond/Maximum_1main/pi/cond/cond_1/pred_id*
T0*:
_output_shapes(
&:         :         
h
gradients/Identity_2Identitygradients/Switch_2:1*
T0*'
_output_shapes
:         
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*
_output_shapes
:
s
gradients/zeros_2/ConstConst^gradients/Identity_2*
dtype0*
_output_shapes
: *
valueB
 *    
Ѕ
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*

index_type0*'
_output_shapes
:         
Л
7gradients/main/pi/cond/cond_1/Pow/Switch_grad/cond_gradMerge?gradients/main/pi/cond/cond_1/Pow_grad/tuple/control_dependencygradients/zeros_2*
T0*
N*)
_output_shapes
:         : 
њ
gradients/Switch_3Switchmain/pi/cond/Maximummain/pi/cond/cond/pred_id*
T0*:
_output_shapes(
&:         :         
h
gradients/Identity_3Identitygradients/Switch_3:1*
T0*'
_output_shapes
:         
e
gradients/Shape_4Shapegradients/Switch_3:1*
_output_shapes
:*
T0*
out_type0
s
gradients/zeros_3/ConstConst^gradients/Identity_3*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѕ
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*

index_type0*'
_output_shapes
:         
═
5gradients/main/pi/cond/cond/Pow/Switch_grad/cond_gradMerge=gradients/main/pi/cond/cond/Pow_grad/tuple/control_dependencygradients/zeros_3*
N*)
_output_shapes
:         : *
T0
г
2gradients/main/q1_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/q1_1/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:ђ
Б
7gradients/main/q1_1/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/q1_1/dense/BiasAdd_grad/BiasAddGrad-^gradients/main/q1_1/dense/Relu_grad/ReluGrad
Д
?gradients/main/q1_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/q1_1/dense/Relu_grad/ReluGrad8^gradients/main/q1_1/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/q1_1/dense/Relu_grad/ReluGrad*(
_output_shapes
:         ђ
е
Agradients/main/q1_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/q1_1/dense/BiasAdd_grad/BiasAddGrad8^gradients/main/q1_1/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/main/q1_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Ј
gradients/AddNAddN7gradients/main/pi/cond/cond_1/Log/Switch_grad/cond_grad7gradients/main/pi/cond/cond_1/Pow/Switch_grad/cond_grad*
T0*J
_class@
><loc:@gradients/main/pi/cond/cond_1/Log/Switch_grad/cond_grad*
N*'
_output_shapes
:         

+gradients/main/pi/cond/Maximum_1_grad/ShapeShapemain/pi/cond/Minimum*
T0*
out_type0*
_output_shapes
:
p
-gradients/main/pi/cond/Maximum_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
{
-gradients/main/pi/cond/Maximum_1_grad/Shape_2Shapegradients/AddN*
T0*
out_type0*
_output_shapes
:
v
1gradients/main/pi/cond/Maximum_1_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
┘
+gradients/main/pi/cond/Maximum_1_grad/zerosFill-gradients/main/pi/cond/Maximum_1_grad/Shape_21gradients/main/pi/cond/Maximum_1_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
ц
2gradients/main/pi/cond/Maximum_1_grad/GreaterEqualGreaterEqualmain/pi/cond/Minimummain/pi/cond/Maximum_1/y*
T0*'
_output_shapes
:         
ь
;gradients/main/pi/cond/Maximum_1_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/main/pi/cond/Maximum_1_grad/Shape-gradients/main/pi/cond/Maximum_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┘
,gradients/main/pi/cond/Maximum_1_grad/SelectSelect2gradients/main/pi/cond/Maximum_1_grad/GreaterEqualgradients/AddN+gradients/main/pi/cond/Maximum_1_grad/zeros*
T0*'
_output_shapes
:         
█
.gradients/main/pi/cond/Maximum_1_grad/Select_1Select2gradients/main/pi/cond/Maximum_1_grad/GreaterEqual+gradients/main/pi/cond/Maximum_1_grad/zerosgradients/AddN*
T0*'
_output_shapes
:         
█
)gradients/main/pi/cond/Maximum_1_grad/SumSum,gradients/main/pi/cond/Maximum_1_grad/Select;gradients/main/pi/cond/Maximum_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
л
-gradients/main/pi/cond/Maximum_1_grad/ReshapeReshape)gradients/main/pi/cond/Maximum_1_grad/Sum+gradients/main/pi/cond/Maximum_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
р
+gradients/main/pi/cond/Maximum_1_grad/Sum_1Sum.gradients/main/pi/cond/Maximum_1_grad/Select_1=gradients/main/pi/cond/Maximum_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┼
/gradients/main/pi/cond/Maximum_1_grad/Reshape_1Reshape+gradients/main/pi/cond/Maximum_1_grad/Sum_1-gradients/main/pi/cond/Maximum_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
а
6gradients/main/pi/cond/Maximum_1_grad/tuple/group_depsNoOp.^gradients/main/pi/cond/Maximum_1_grad/Reshape0^gradients/main/pi/cond/Maximum_1_grad/Reshape_1
д
>gradients/main/pi/cond/Maximum_1_grad/tuple/control_dependencyIdentity-gradients/main/pi/cond/Maximum_1_grad/Reshape7^gradients/main/pi/cond/Maximum_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/pi/cond/Maximum_1_grad/Reshape*'
_output_shapes
:         
Џ
@gradients/main/pi/cond/Maximum_1_grad/tuple/control_dependency_1Identity/gradients/main/pi/cond/Maximum_1_grad/Reshape_17^gradients/main/pi/cond/Maximum_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/pi/cond/Maximum_1_grad/Reshape_1*
_output_shapes
: 
І
gradients/AddN_1AddN5gradients/main/pi/cond/cond/Log/Switch_grad/cond_grad5gradients/main/pi/cond/cond/Pow/Switch_grad/cond_grad*
T0*H
_class>
<:loc:@gradients/main/pi/cond/cond/Log/Switch_grad/cond_grad*
N*'
_output_shapes
:         
y
)gradients/main/pi/cond/Maximum_grad/ShapeShapemain/pi/cond/Exp*
T0*
out_type0*
_output_shapes
:
n
+gradients/main/pi/cond/Maximum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
{
+gradients/main/pi/cond/Maximum_grad/Shape_2Shapegradients/AddN_1*
T0*
out_type0*
_output_shapes
:
t
/gradients/main/pi/cond/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
М
)gradients/main/pi/cond/Maximum_grad/zerosFill+gradients/main/pi/cond/Maximum_grad/Shape_2/gradients/main/pi/cond/Maximum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
ю
0gradients/main/pi/cond/Maximum_grad/GreaterEqualGreaterEqualmain/pi/cond/Expmain/pi/cond/Maximum/y*
T0*'
_output_shapes
:         
у
9gradients/main/pi/cond/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/main/pi/cond/Maximum_grad/Shape+gradients/main/pi/cond/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Н
*gradients/main/pi/cond/Maximum_grad/SelectSelect0gradients/main/pi/cond/Maximum_grad/GreaterEqualgradients/AddN_1)gradients/main/pi/cond/Maximum_grad/zeros*
T0*'
_output_shapes
:         
О
,gradients/main/pi/cond/Maximum_grad/Select_1Select0gradients/main/pi/cond/Maximum_grad/GreaterEqual)gradients/main/pi/cond/Maximum_grad/zerosgradients/AddN_1*'
_output_shapes
:         *
T0
Н
'gradients/main/pi/cond/Maximum_grad/SumSum*gradients/main/pi/cond/Maximum_grad/Select9gradients/main/pi/cond/Maximum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╩
+gradients/main/pi/cond/Maximum_grad/ReshapeReshape'gradients/main/pi/cond/Maximum_grad/Sum)gradients/main/pi/cond/Maximum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
█
)gradients/main/pi/cond/Maximum_grad/Sum_1Sum,gradients/main/pi/cond/Maximum_grad/Select_1;gradients/main/pi/cond/Maximum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┐
-gradients/main/pi/cond/Maximum_grad/Reshape_1Reshape)gradients/main/pi/cond/Maximum_grad/Sum_1+gradients/main/pi/cond/Maximum_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
џ
4gradients/main/pi/cond/Maximum_grad/tuple/group_depsNoOp,^gradients/main/pi/cond/Maximum_grad/Reshape.^gradients/main/pi/cond/Maximum_grad/Reshape_1
ъ
<gradients/main/pi/cond/Maximum_grad/tuple/control_dependencyIdentity+gradients/main/pi/cond/Maximum_grad/Reshape5^gradients/main/pi/cond/Maximum_grad/tuple/group_deps*'
_output_shapes
:         *
T0*>
_class4
20loc:@gradients/main/pi/cond/Maximum_grad/Reshape
Њ
>gradients/main/pi/cond/Maximum_grad/tuple/control_dependency_1Identity-gradients/main/pi/cond/Maximum_grad/Reshape_15^gradients/main/pi/cond/Maximum_grad/tuple/group_deps*
_output_shapes
: *
T0*@
_class6
42loc:@gradients/main/pi/cond/Maximum_grad/Reshape_1
Ж
,gradients/main/q1_1/dense/MatMul_grad/MatMulMatMul?gradients/main/q1_1/dense/BiasAdd_grad/tuple/control_dependencymain/q1/dense/kernel/read*
T0*'
_output_shapes
:         h*
transpose_a( *
transpose_b(
█
.gradients/main/q1_1/dense/MatMul_grad/MatMul_1MatMulmain/q1_1/concat?gradients/main/q1_1/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	hђ*
transpose_a(*
transpose_b( 
ъ
6gradients/main/q1_1/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/main/q1_1/dense/MatMul_grad/MatMul/^gradients/main/q1_1/dense/MatMul_grad/MatMul_1
ц
>gradients/main/q1_1/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/q1_1/dense/MatMul_grad/MatMul7^gradients/main/q1_1/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:         h*
T0*?
_class5
31loc:@gradients/main/q1_1/dense/MatMul_grad/MatMul
б
@gradients/main/q1_1/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/q1_1/dense/MatMul_grad/MatMul_17^gradients/main/q1_1/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/q1_1/dense/MatMul_grad/MatMul_1*
_output_shapes
:	hђ
{
)gradients/main/pi/cond/Minimum_grad/ShapeShapemain/pi/cond/Exp_1*
T0*
out_type0*
_output_shapes
:
n
+gradients/main/pi/cond/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Е
+gradients/main/pi/cond/Minimum_grad/Shape_2Shape>gradients/main/pi/cond/Maximum_1_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
t
/gradients/main/pi/cond/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
М
)gradients/main/pi/cond/Minimum_grad/zerosFill+gradients/main/pi/cond/Minimum_grad/Shape_2/gradients/main/pi/cond/Minimum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
њ
-gradients/main/pi/cond/Minimum_grad/LessEqual	LessEqualmain/pi/cond/Exp_1main/pi/cond/Pow*
T0*'
_output_shapes
:         
у
9gradients/main/pi/cond/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/main/pi/cond/Minimum_grad/Shape+gradients/main/pi/cond/Minimum_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ђ
*gradients/main/pi/cond/Minimum_grad/SelectSelect-gradients/main/pi/cond/Minimum_grad/LessEqual>gradients/main/pi/cond/Maximum_1_grad/tuple/control_dependency)gradients/main/pi/cond/Minimum_grad/zeros*
T0*'
_output_shapes
:         
ѓ
,gradients/main/pi/cond/Minimum_grad/Select_1Select-gradients/main/pi/cond/Minimum_grad/LessEqual)gradients/main/pi/cond/Minimum_grad/zeros>gradients/main/pi/cond/Maximum_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
Н
'gradients/main/pi/cond/Minimum_grad/SumSum*gradients/main/pi/cond/Minimum_grad/Select9gradients/main/pi/cond/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╩
+gradients/main/pi/cond/Minimum_grad/ReshapeReshape'gradients/main/pi/cond/Minimum_grad/Sum)gradients/main/pi/cond/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
█
)gradients/main/pi/cond/Minimum_grad/Sum_1Sum,gradients/main/pi/cond/Minimum_grad/Select_1;gradients/main/pi/cond/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┐
-gradients/main/pi/cond/Minimum_grad/Reshape_1Reshape)gradients/main/pi/cond/Minimum_grad/Sum_1+gradients/main/pi/cond/Minimum_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
џ
4gradients/main/pi/cond/Minimum_grad/tuple/group_depsNoOp,^gradients/main/pi/cond/Minimum_grad/Reshape.^gradients/main/pi/cond/Minimum_grad/Reshape_1
ъ
<gradients/main/pi/cond/Minimum_grad/tuple/control_dependencyIdentity+gradients/main/pi/cond/Minimum_grad/Reshape5^gradients/main/pi/cond/Minimum_grad/tuple/group_deps*'
_output_shapes
:         *
T0*>
_class4
20loc:@gradients/main/pi/cond/Minimum_grad/Reshape
Њ
>gradients/main/pi/cond/Minimum_grad/tuple/control_dependency_1Identity-gradients/main/pi/cond/Minimum_grad/Reshape_15^gradients/main/pi/cond/Minimum_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/pi/cond/Minimum_grad/Reshape_1*
_output_shapes
: 
г
#gradients/main/pi/cond/Exp_grad/mulMul<gradients/main/pi/cond/Maximum_grad/tuple/control_dependencymain/pi/cond/Exp*
T0*'
_output_shapes
:         
f
$gradients/main/q1_1/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Ї
#gradients/main/q1_1/concat_grad/modFloorModmain/q1_1/concat/axis$gradients/main/q1_1/concat_grad/Rank*
_output_shapes
: *
T0
p
%gradients/main/q1_1/concat_grad/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
Ї
&gradients/main/q1_1/concat_grad/ShapeNShapeNPlaceholder
main/mul_1*
T0*
out_type0*
N* 
_output_shapes
::
я
,gradients/main/q1_1/concat_grad/ConcatOffsetConcatOffset#gradients/main/q1_1/concat_grad/mod&gradients/main/q1_1/concat_grad/ShapeN(gradients/main/q1_1/concat_grad/ShapeN:1*
N* 
_output_shapes
::
Ѓ
%gradients/main/q1_1/concat_grad/SliceSlice>gradients/main/q1_1/dense/MatMul_grad/tuple/control_dependency,gradients/main/q1_1/concat_grad/ConcatOffset&gradients/main/q1_1/concat_grad/ShapeN*'
_output_shapes
:         I*
Index0*
T0
Ѕ
'gradients/main/q1_1/concat_grad/Slice_1Slice>gradients/main/q1_1/dense/MatMul_grad/tuple/control_dependency.gradients/main/q1_1/concat_grad/ConcatOffset:1(gradients/main/q1_1/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:         
і
0gradients/main/q1_1/concat_grad/tuple/group_depsNoOp&^gradients/main/q1_1/concat_grad/Slice(^gradients/main/q1_1/concat_grad/Slice_1
і
8gradients/main/q1_1/concat_grad/tuple/control_dependencyIdentity%gradients/main/q1_1/concat_grad/Slice1^gradients/main/q1_1/concat_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/main/q1_1/concat_grad/Slice*'
_output_shapes
:         I
љ
:gradients/main/q1_1/concat_grad/tuple/control_dependency_1Identity'gradients/main/q1_1/concat_grad/Slice_11^gradients/main/q1_1/concat_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/q1_1/concat_grad/Slice_1*'
_output_shapes
:         
░
%gradients/main/pi/cond/Exp_1_grad/mulMul<gradients/main/pi/cond/Minimum_grad/tuple/control_dependencymain/pi/cond/Exp_1*
T0*'
_output_shapes
:         
є
gradients/Switch_4Switchmain/pi/sub_5main/pi/cond/pred_id*
T0*:
_output_shapes(
&:         :         
f
gradients/Identity_4Identitygradients/Switch_4*
T0*'
_output_shapes
:         
c
gradients/Shape_5Shapegradients/Switch_4*
T0*
out_type0*
_output_shapes
:
s
gradients/zeros_4/ConstConst^gradients/Identity_4*
dtype0*
_output_shapes
: *
valueB
 *    
Ѕ
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*'
_output_shapes
:         *
T0*

index_type0
«
0gradients/main/pi/cond/Exp/Switch_grad/cond_gradMergegradients/zeros_4#gradients/main/pi/cond/Exp_grad/mul*
T0*
N*)
_output_shapes
:         : 
m
gradients/main/mul_1_grad/ShapeShapemain/pi/Tanh_1*
T0*
out_type0*
_output_shapes
:
d
!gradients/main/mul_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╔
/gradients/main/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/main/mul_1_grad/Shape!gradients/main/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
а
gradients/main/mul_1_grad/MulMul:gradients/main/q1_1/concat_grad/tuple/control_dependency_1main/mul_1/y*
T0*'
_output_shapes
:         
┤
gradients/main/mul_1_grad/SumSumgradients/main/mul_1_grad/Mul/gradients/main/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
г
!gradients/main/mul_1_grad/ReshapeReshapegradients/main/mul_1_grad/Sumgradients/main/mul_1_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
ц
gradients/main/mul_1_grad/Mul_1Mulmain/pi/Tanh_1:gradients/main/q1_1/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
║
gradients/main/mul_1_grad/Sum_1Sumgradients/main/mul_1_grad/Mul_11gradients/main/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
А
#gradients/main/mul_1_grad/Reshape_1Reshapegradients/main/mul_1_grad/Sum_1!gradients/main/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/main/mul_1_grad/tuple/group_depsNoOp"^gradients/main/mul_1_grad/Reshape$^gradients/main/mul_1_grad/Reshape_1
Ш
2gradients/main/mul_1_grad/tuple/control_dependencyIdentity!gradients/main/mul_1_grad/Reshape+^gradients/main/mul_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/main/mul_1_grad/Reshape*'
_output_shapes
:         
в
4gradients/main/mul_1_grad/tuple/control_dependency_1Identity#gradients/main/mul_1_grad/Reshape_1+^gradients/main/mul_1_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/main/mul_1_grad/Reshape_1*
_output_shapes
: 
є
gradients/Switch_5Switchmain/pi/sub_5main/pi/cond/pred_id*
T0*:
_output_shapes(
&:         :         
h
gradients/Identity_5Identitygradients/Switch_5:1*
T0*'
_output_shapes
:         
e
gradients/Shape_6Shapegradients/Switch_5:1*
T0*
out_type0*
_output_shapes
:
s
gradients/zeros_5/ConstConst^gradients/Identity_5*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѕ
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*

index_type0*'
_output_shapes
:         
▓
2gradients/main/pi/cond/Exp_1/Switch_grad/cond_gradMerge%gradients/main/pi/cond/Exp_1_grad/mulgradients/zeros_5*
T0*
N*)
_output_shapes
:         : 
■
gradients/AddN_2AddN0gradients/main/pi/cond/Exp/Switch_grad/cond_grad2gradients/main/pi/cond/Exp_1/Switch_grad/cond_grad*
T0*C
_class9
75loc:@gradients/main/pi/cond/Exp/Switch_grad/cond_grad*
N*'
_output_shapes
:         
o
"gradients/main/pi/sub_5_grad/ShapeShapemain/pi/mul_3*
T0*
out_type0*
_output_shapes
:
o
$gradients/main/pi/sub_5_grad/Shape_1Shapemain/pi/Log*
_output_shapes
:*
T0*
out_type0
м
2gradients/main/pi/sub_5_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/sub_5_grad/Shape$gradients/main/pi/sub_5_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Г
 gradients/main/pi/sub_5_grad/SumSumgradients/AddN_22gradients/main/pi/sub_5_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
$gradients/main/pi/sub_5_grad/ReshapeReshape gradients/main/pi/sub_5_grad/Sum"gradients/main/pi/sub_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
▒
"gradients/main/pi/sub_5_grad/Sum_1Sumgradients/AddN_24gradients/main/pi/sub_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
n
 gradients/main/pi/sub_5_grad/NegNeg"gradients/main/pi/sub_5_grad/Sum_1*
_output_shapes
:*
T0
╣
&gradients/main/pi/sub_5_grad/Reshape_1Reshape gradients/main/pi/sub_5_grad/Neg$gradients/main/pi/sub_5_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ё
-gradients/main/pi/sub_5_grad/tuple/group_depsNoOp%^gradients/main/pi/sub_5_grad/Reshape'^gradients/main/pi/sub_5_grad/Reshape_1
ѓ
5gradients/main/pi/sub_5_grad/tuple/control_dependencyIdentity$gradients/main/pi/sub_5_grad/Reshape.^gradients/main/pi/sub_5_grad/tuple/group_deps*'
_output_shapes
:         *
T0*7
_class-
+)loc:@gradients/main/pi/sub_5_grad/Reshape
ѕ
7gradients/main/pi/sub_5_grad/tuple/control_dependency_1Identity&gradients/main/pi/sub_5_grad/Reshape_1.^gradients/main/pi/sub_5_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/sub_5_grad/Reshape_1*'
_output_shapes
:         
e
"gradients/main/pi/mul_3_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
q
$gradients/main/pi/mul_3_grad/Shape_1Shapemain/pi/add_5*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/mul_3_grad/Shape$gradients/main/pi/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ъ
 gradients/main/pi/mul_3_grad/MulMul5gradients/main/pi/sub_5_grad/tuple/control_dependencymain/pi/add_5*'
_output_shapes
:         *
T0
й
 gradients/main/pi/mul_3_grad/SumSum gradients/main/pi/mul_3_grad/Mul2gradients/main/pi/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ц
$gradients/main/pi/mul_3_grad/ReshapeReshape gradients/main/pi/mul_3_grad/Sum"gradients/main/pi/mul_3_grad/Shape*
_output_shapes
: *
T0*
Tshape0
Б
"gradients/main/pi/mul_3_grad/Mul_1Mulmain/pi/mul_3/x5gradients/main/pi/sub_5_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
├
"gradients/main/pi/mul_3_grad/Sum_1Sum"gradients/main/pi/mul_3_grad/Mul_14gradients/main/pi/mul_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╗
&gradients/main/pi/mul_3_grad/Reshape_1Reshape"gradients/main/pi/mul_3_grad/Sum_1$gradients/main/pi/mul_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ё
-gradients/main/pi/mul_3_grad/tuple/group_depsNoOp%^gradients/main/pi/mul_3_grad/Reshape'^gradients/main/pi/mul_3_grad/Reshape_1
ы
5gradients/main/pi/mul_3_grad/tuple/control_dependencyIdentity$gradients/main/pi/mul_3_grad/Reshape.^gradients/main/pi/mul_3_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/mul_3_grad/Reshape*
_output_shapes
: 
ѕ
7gradients/main/pi/mul_3_grad/tuple/control_dependency_1Identity&gradients/main/pi/mul_3_grad/Reshape_1.^gradients/main/pi/mul_3_grad/tuple/group_deps*'
_output_shapes
:         *
T0*9
_class/
-+loc:@gradients/main/pi/mul_3_grad/Reshape_1
«
%gradients/main/pi/Log_grad/Reciprocal
Reciprocalmain/pi/add_88^gradients/main/pi/sub_5_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
и
gradients/main/pi/Log_grad/mulMul7gradients/main/pi/sub_5_grad/tuple/control_dependency_1%gradients/main/pi/Log_grad/Reciprocal*'
_output_shapes
:         *
T0
o
"gradients/main/pi/add_5_grad/ShapeShapemain/pi/add_4*
_output_shapes
:*
T0*
out_type0
g
$gradients/main/pi/add_5_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
м
2gradients/main/pi/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/add_5_grad/Shape$gradients/main/pi/add_5_grad/Shape_1*
T0*2
_output_shapes 
:         :         
н
 gradients/main/pi/add_5_grad/SumSum7gradients/main/pi/mul_3_grad/tuple/control_dependency_12gradients/main/pi/add_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
$gradients/main/pi/add_5_grad/ReshapeReshape gradients/main/pi/add_5_grad/Sum"gradients/main/pi/add_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
п
"gradients/main/pi/add_5_grad/Sum_1Sum7gradients/main/pi/mul_3_grad/tuple/control_dependency_14gradients/main/pi/add_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ф
&gradients/main/pi/add_5_grad/Reshape_1Reshape"gradients/main/pi/add_5_grad/Sum_1$gradients/main/pi/add_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ё
-gradients/main/pi/add_5_grad/tuple/group_depsNoOp%^gradients/main/pi/add_5_grad/Reshape'^gradients/main/pi/add_5_grad/Reshape_1
ѓ
5gradients/main/pi/add_5_grad/tuple/control_dependencyIdentity$gradients/main/pi/add_5_grad/Reshape.^gradients/main/pi/add_5_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/add_5_grad/Reshape*'
_output_shapes
:         
э
7gradients/main/pi/add_5_grad/tuple/control_dependency_1Identity&gradients/main/pi/add_5_grad/Reshape_1.^gradients/main/pi/add_5_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/add_5_grad/Reshape_1*
_output_shapes
: 
o
"gradients/main/pi/add_8_grad/ShapeShapemain/pi/add_7*
T0*
out_type0*
_output_shapes
:
g
$gradients/main/pi/add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
м
2gradients/main/pi/add_8_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/add_8_grad/Shape$gradients/main/pi/add_8_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╗
 gradients/main/pi/add_8_grad/SumSumgradients/main/pi/Log_grad/mul2gradients/main/pi/add_8_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
$gradients/main/pi/add_8_grad/ReshapeReshape gradients/main/pi/add_8_grad/Sum"gradients/main/pi/add_8_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
┐
"gradients/main/pi/add_8_grad/Sum_1Sumgradients/main/pi/Log_grad/mul4gradients/main/pi/add_8_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ф
&gradients/main/pi/add_8_grad/Reshape_1Reshape"gradients/main/pi/add_8_grad/Sum_1$gradients/main/pi/add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ё
-gradients/main/pi/add_8_grad/tuple/group_depsNoOp%^gradients/main/pi/add_8_grad/Reshape'^gradients/main/pi/add_8_grad/Reshape_1
ѓ
5gradients/main/pi/add_8_grad/tuple/control_dependencyIdentity$gradients/main/pi/add_8_grad/Reshape.^gradients/main/pi/add_8_grad/tuple/group_deps*'
_output_shapes
:         *
T0*7
_class-
+)loc:@gradients/main/pi/add_8_grad/Reshape
э
7gradients/main/pi/add_8_grad/tuple/control_dependency_1Identity&gradients/main/pi/add_8_grad/Reshape_1.^gradients/main/pi/add_8_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/add_8_grad/Reshape_1*
_output_shapes
: 
m
"gradients/main/pi/add_4_grad/ShapeShapemain/pi/pow*
_output_shapes
:*
T0*
out_type0
q
$gradients/main/pi/add_4_grad/Shape_1Shapemain/pi/mul_2*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/add_4_grad/Shape$gradients/main/pi/add_4_grad/Shape_1*
T0*2
_output_shapes 
:         :         
м
 gradients/main/pi/add_4_grad/SumSum5gradients/main/pi/add_5_grad/tuple/control_dependency2gradients/main/pi/add_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
$gradients/main/pi/add_4_grad/ReshapeReshape gradients/main/pi/add_4_grad/Sum"gradients/main/pi/add_4_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
о
"gradients/main/pi/add_4_grad/Sum_1Sum5gradients/main/pi/add_5_grad/tuple/control_dependency4gradients/main/pi/add_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╗
&gradients/main/pi/add_4_grad/Reshape_1Reshape"gradients/main/pi/add_4_grad/Sum_1$gradients/main/pi/add_4_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
Ё
-gradients/main/pi/add_4_grad/tuple/group_depsNoOp%^gradients/main/pi/add_4_grad/Reshape'^gradients/main/pi/add_4_grad/Reshape_1
ѓ
5gradients/main/pi/add_4_grad/tuple/control_dependencyIdentity$gradients/main/pi/add_4_grad/Reshape.^gradients/main/pi/add_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/add_4_grad/Reshape*'
_output_shapes
:         
ѕ
7gradients/main/pi/add_4_grad/tuple/control_dependency_1Identity&gradients/main/pi/add_4_grad/Reshape_1.^gradients/main/pi/add_4_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/add_4_grad/Reshape_1*'
_output_shapes
:         
o
"gradients/main/pi/add_7_grad/ShapeShapemain/pi/sub_2*
_output_shapes
:*
T0*
out_type0
x
$gradients/main/pi/add_7_grad/Shape_1Shapemain/pi/StopGradient*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/add_7_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/add_7_grad/Shape$gradients/main/pi/add_7_grad/Shape_1*
T0*2
_output_shapes 
:         :         
м
 gradients/main/pi/add_7_grad/SumSum5gradients/main/pi/add_8_grad/tuple/control_dependency2gradients/main/pi/add_7_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
х
$gradients/main/pi/add_7_grad/ReshapeReshape gradients/main/pi/add_7_grad/Sum"gradients/main/pi/add_7_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
о
"gradients/main/pi/add_7_grad/Sum_1Sum5gradients/main/pi/add_8_grad/tuple/control_dependency4gradients/main/pi/add_7_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╗
&gradients/main/pi/add_7_grad/Reshape_1Reshape"gradients/main/pi/add_7_grad/Sum_1$gradients/main/pi/add_7_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
Ё
-gradients/main/pi/add_7_grad/tuple/group_depsNoOp%^gradients/main/pi/add_7_grad/Reshape'^gradients/main/pi/add_7_grad/Reshape_1
ѓ
5gradients/main/pi/add_7_grad/tuple/control_dependencyIdentity$gradients/main/pi/add_7_grad/Reshape.^gradients/main/pi/add_7_grad/tuple/group_deps*'
_output_shapes
:         *
T0*7
_class-
+)loc:@gradients/main/pi/add_7_grad/Reshape
ѕ
7gradients/main/pi/add_7_grad/tuple/control_dependency_1Identity&gradients/main/pi/add_7_grad/Reshape_1.^gradients/main/pi/add_7_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/add_7_grad/Reshape_1*'
_output_shapes
:         
o
 gradients/main/pi/pow_grad/ShapeShapemain/pi/truediv*
T0*
out_type0*
_output_shapes
:
e
"gradients/main/pi/pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╠
0gradients/main/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/main/pi/pow_grad/Shape"gradients/main/pi/pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ю
gradients/main/pi/pow_grad/mulMul5gradients/main/pi/add_4_grad/tuple/control_dependencymain/pi/pow/y*
T0*'
_output_shapes
:         
e
 gradients/main/pi/pow_grad/sub/yConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
w
gradients/main/pi/pow_grad/subSubmain/pi/pow/y gradients/main/pi/pow_grad/sub/y*
_output_shapes
: *
T0
ѕ
gradients/main/pi/pow_grad/PowPowmain/pi/truedivgradients/main/pi/pow_grad/sub*'
_output_shapes
:         *
T0
Ў
 gradients/main/pi/pow_grad/mul_1Mulgradients/main/pi/pow_grad/mulgradients/main/pi/pow_grad/Pow*
T0*'
_output_shapes
:         
╣
gradients/main/pi/pow_grad/SumSum gradients/main/pi/pow_grad/mul_10gradients/main/pi/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
»
"gradients/main/pi/pow_grad/ReshapeReshapegradients/main/pi/pow_grad/Sum gradients/main/pi/pow_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
i
$gradients/main/pi/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ќ
"gradients/main/pi/pow_grad/GreaterGreatermain/pi/truediv$gradients/main/pi/pow_grad/Greater/y*'
_output_shapes
:         *
T0
y
*gradients/main/pi/pow_grad/ones_like/ShapeShapemain/pi/truediv*
T0*
out_type0*
_output_shapes
:
o
*gradients/main/pi/pow_grad/ones_like/ConstConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╚
$gradients/main/pi/pow_grad/ones_likeFill*gradients/main/pi/pow_grad/ones_like/Shape*gradients/main/pi/pow_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:         
И
!gradients/main/pi/pow_grad/SelectSelect"gradients/main/pi/pow_grad/Greatermain/pi/truediv$gradients/main/pi/pow_grad/ones_like*
T0*'
_output_shapes
:         
z
gradients/main/pi/pow_grad/LogLog!gradients/main/pi/pow_grad/Select*
T0*'
_output_shapes
:         
u
%gradients/main/pi/pow_grad/zeros_like	ZerosLikemain/pi/truediv*
T0*'
_output_shapes
:         
╩
#gradients/main/pi/pow_grad/Select_1Select"gradients/main/pi/pow_grad/Greatergradients/main/pi/pow_grad/Log%gradients/main/pi/pow_grad/zeros_like*
T0*'
_output_shapes
:         
Ю
 gradients/main/pi/pow_grad/mul_2Mul5gradients/main/pi/add_4_grad/tuple/control_dependencymain/pi/pow*
T0*'
_output_shapes
:         
а
 gradients/main/pi/pow_grad/mul_3Mul gradients/main/pi/pow_grad/mul_2#gradients/main/pi/pow_grad/Select_1*'
_output_shapes
:         *
T0
й
 gradients/main/pi/pow_grad/Sum_1Sum gradients/main/pi/pow_grad/mul_32gradients/main/pi/pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ц
$gradients/main/pi/pow_grad/Reshape_1Reshape gradients/main/pi/pow_grad/Sum_1"gradients/main/pi/pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

+gradients/main/pi/pow_grad/tuple/group_depsNoOp#^gradients/main/pi/pow_grad/Reshape%^gradients/main/pi/pow_grad/Reshape_1
Щ
3gradients/main/pi/pow_grad/tuple/control_dependencyIdentity"gradients/main/pi/pow_grad/Reshape,^gradients/main/pi/pow_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/main/pi/pow_grad/Reshape*'
_output_shapes
:         
№
5gradients/main/pi/pow_grad/tuple/control_dependency_1Identity$gradients/main/pi/pow_grad/Reshape_1,^gradients/main/pi/pow_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/pow_grad/Reshape_1*
_output_shapes
: 
e
"gradients/main/pi/mul_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
q
$gradients/main/pi/mul_2_grad/Shape_1Shapemain/pi/add_1*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/mul_2_grad/Shape$gradients/main/pi/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
А
 gradients/main/pi/mul_2_grad/MulMul7gradients/main/pi/add_4_grad/tuple/control_dependency_1main/pi/add_1*
T0*'
_output_shapes
:         
й
 gradients/main/pi/mul_2_grad/SumSum gradients/main/pi/mul_2_grad/Mul2gradients/main/pi/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ц
$gradients/main/pi/mul_2_grad/ReshapeReshape gradients/main/pi/mul_2_grad/Sum"gradients/main/pi/mul_2_grad/Shape*
_output_shapes
: *
T0*
Tshape0
Ц
"gradients/main/pi/mul_2_grad/Mul_1Mulmain/pi/mul_2/x7gradients/main/pi/add_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
├
"gradients/main/pi/mul_2_grad/Sum_1Sum"gradients/main/pi/mul_2_grad/Mul_14gradients/main/pi/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╗
&gradients/main/pi/mul_2_grad/Reshape_1Reshape"gradients/main/pi/mul_2_grad/Sum_1$gradients/main/pi/mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ё
-gradients/main/pi/mul_2_grad/tuple/group_depsNoOp%^gradients/main/pi/mul_2_grad/Reshape'^gradients/main/pi/mul_2_grad/Reshape_1
ы
5gradients/main/pi/mul_2_grad/tuple/control_dependencyIdentity$gradients/main/pi/mul_2_grad/Reshape.^gradients/main/pi/mul_2_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/main/pi/mul_2_grad/Reshape
ѕ
7gradients/main/pi/mul_2_grad/tuple/control_dependency_1Identity&gradients/main/pi/mul_2_grad/Reshape_1.^gradients/main/pi/mul_2_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/mul_2_grad/Reshape_1*'
_output_shapes
:         
q
$gradients/main/pi/truediv_grad/ShapeShapemain/pi/sub_1*
_output_shapes
:*
T0*
out_type0
s
&gradients/main/pi/truediv_grad/Shape_1Shapemain/pi/add_3*
T0*
out_type0*
_output_shapes
:
п
4gradients/main/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/main/pi/truediv_grad/Shape&gradients/main/pi/truediv_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Д
&gradients/main/pi/truediv_grad/RealDivRealDiv3gradients/main/pi/pow_grad/tuple/control_dependencymain/pi/add_3*'
_output_shapes
:         *
T0
К
"gradients/main/pi/truediv_grad/SumSum&gradients/main/pi/truediv_grad/RealDiv4gradients/main/pi/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╗
&gradients/main/pi/truediv_grad/ReshapeReshape"gradients/main/pi/truediv_grad/Sum$gradients/main/pi/truediv_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
j
"gradients/main/pi/truediv_grad/NegNegmain/pi/sub_1*
T0*'
_output_shapes
:         
ў
(gradients/main/pi/truediv_grad/RealDiv_1RealDiv"gradients/main/pi/truediv_grad/Negmain/pi/add_3*
T0*'
_output_shapes
:         
ъ
(gradients/main/pi/truediv_grad/RealDiv_2RealDiv(gradients/main/pi/truediv_grad/RealDiv_1main/pi/add_3*
T0*'
_output_shapes
:         
║
"gradients/main/pi/truediv_grad/mulMul3gradients/main/pi/pow_grad/tuple/control_dependency(gradients/main/pi/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:         
К
$gradients/main/pi/truediv_grad/Sum_1Sum"gradients/main/pi/truediv_grad/mul6gradients/main/pi/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┴
(gradients/main/pi/truediv_grad/Reshape_1Reshape$gradients/main/pi/truediv_grad/Sum_1&gradients/main/pi/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
І
/gradients/main/pi/truediv_grad/tuple/group_depsNoOp'^gradients/main/pi/truediv_grad/Reshape)^gradients/main/pi/truediv_grad/Reshape_1
і
7gradients/main/pi/truediv_grad/tuple/control_dependencyIdentity&gradients/main/pi/truediv_grad/Reshape0^gradients/main/pi/truediv_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/truediv_grad/Reshape*'
_output_shapes
:         
љ
9gradients/main/pi/truediv_grad/tuple/control_dependency_1Identity(gradients/main/pi/truediv_grad/Reshape_10^gradients/main/pi/truediv_grad/tuple/group_deps*'
_output_shapes
:         *
T0*;
_class1
/-loc:@gradients/main/pi/truediv_grad/Reshape_1
o
"gradients/main/pi/sub_1_grad/ShapeShapemain/pi/add_2*
T0*
out_type0*
_output_shapes
:
{
$gradients/main/pi/sub_1_grad/Shape_1Shapemain/pi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/sub_1_grad/Shape$gradients/main/pi/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
н
 gradients/main/pi/sub_1_grad/SumSum7gradients/main/pi/truediv_grad/tuple/control_dependency2gradients/main/pi/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
$gradients/main/pi/sub_1_grad/ReshapeReshape gradients/main/pi/sub_1_grad/Sum"gradients/main/pi/sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
п
"gradients/main/pi/sub_1_grad/Sum_1Sum7gradients/main/pi/truediv_grad/tuple/control_dependency4gradients/main/pi/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
n
 gradients/main/pi/sub_1_grad/NegNeg"gradients/main/pi/sub_1_grad/Sum_1*
_output_shapes
:*
T0
╣
&gradients/main/pi/sub_1_grad/Reshape_1Reshape gradients/main/pi/sub_1_grad/Neg$gradients/main/pi/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ё
-gradients/main/pi/sub_1_grad/tuple/group_depsNoOp%^gradients/main/pi/sub_1_grad/Reshape'^gradients/main/pi/sub_1_grad/Reshape_1
ѓ
5gradients/main/pi/sub_1_grad/tuple/control_dependencyIdentity$gradients/main/pi/sub_1_grad/Reshape.^gradients/main/pi/sub_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/sub_1_grad/Reshape*'
_output_shapes
:         
ѕ
7gradients/main/pi/sub_1_grad/tuple/control_dependency_1Identity&gradients/main/pi/sub_1_grad/Reshape_1.^gradients/main/pi/sub_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/sub_1_grad/Reshape_1*'
_output_shapes
:         
o
"gradients/main/pi/add_3_grad/ShapeShapemain/pi/Exp_1*
T0*
out_type0*
_output_shapes
:
g
$gradients/main/pi/add_3_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
м
2gradients/main/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/add_3_grad/Shape$gradients/main/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
о
 gradients/main/pi/add_3_grad/SumSum9gradients/main/pi/truediv_grad/tuple/control_dependency_12gradients/main/pi/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
$gradients/main/pi/add_3_grad/ReshapeReshape gradients/main/pi/add_3_grad/Sum"gradients/main/pi/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
┌
"gradients/main/pi/add_3_grad/Sum_1Sum9gradients/main/pi/truediv_grad/tuple/control_dependency_14gradients/main/pi/add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ф
&gradients/main/pi/add_3_grad/Reshape_1Reshape"gradients/main/pi/add_3_grad/Sum_1$gradients/main/pi/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ё
-gradients/main/pi/add_3_grad/tuple/group_depsNoOp%^gradients/main/pi/add_3_grad/Reshape'^gradients/main/pi/add_3_grad/Reshape_1
ѓ
5gradients/main/pi/add_3_grad/tuple/control_dependencyIdentity$gradients/main/pi/add_3_grad/Reshape.^gradients/main/pi/add_3_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/add_3_grad/Reshape*'
_output_shapes
:         
э
7gradients/main/pi/add_3_grad/tuple/control_dependency_1Identity&gradients/main/pi/add_3_grad/Reshape_1.^gradients/main/pi/add_3_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/add_3_grad/Reshape_1*
_output_shapes
: 
Ъ
 gradients/main/pi/Exp_1_grad/mulMul5gradients/main/pi/add_3_grad/tuple/control_dependencymain/pi/Exp_1*
T0*'
_output_shapes
:         
e
"gradients/main/pi/sub_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
q
$gradients/main/pi/sub_2_grad/Shape_1Shapemain/pi/pow_1*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/sub_2_grad/Shape$gradients/main/pi/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
м
 gradients/main/pi/sub_2_grad/SumSum5gradients/main/pi/add_7_grad/tuple/control_dependency2gradients/main/pi/sub_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ц
$gradients/main/pi/sub_2_grad/ReshapeReshape gradients/main/pi/sub_2_grad/Sum"gradients/main/pi/sub_2_grad/Shape*
_output_shapes
: *
T0*
Tshape0
о
"gradients/main/pi/sub_2_grad/Sum_1Sum5gradients/main/pi/add_7_grad/tuple/control_dependency4gradients/main/pi/sub_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
n
 gradients/main/pi/sub_2_grad/NegNeg"gradients/main/pi/sub_2_grad/Sum_1*
T0*
_output_shapes
:
╣
&gradients/main/pi/sub_2_grad/Reshape_1Reshape gradients/main/pi/sub_2_grad/Neg$gradients/main/pi/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ё
-gradients/main/pi/sub_2_grad/tuple/group_depsNoOp%^gradients/main/pi/sub_2_grad/Reshape'^gradients/main/pi/sub_2_grad/Reshape_1
ы
5gradients/main/pi/sub_2_grad/tuple/control_dependencyIdentity$gradients/main/pi/sub_2_grad/Reshape.^gradients/main/pi/sub_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/sub_2_grad/Reshape*
_output_shapes
: 
ѕ
7gradients/main/pi/sub_2_grad/tuple/control_dependency_1Identity&gradients/main/pi/sub_2_grad/Reshape_1.^gradients/main/pi/sub_2_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/sub_2_grad/Reshape_1*'
_output_shapes
:         
p
"gradients/main/pi/pow_1_grad/ShapeShapemain/pi/Tanh_1*
T0*
out_type0*
_output_shapes
:
g
$gradients/main/pi/pow_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
м
2gradients/main/pi/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/pow_1_grad/Shape$gradients/main/pi/pow_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Б
 gradients/main/pi/pow_1_grad/mulMul7gradients/main/pi/sub_2_grad/tuple/control_dependency_1main/pi/pow_1/y*'
_output_shapes
:         *
T0
g
"gradients/main/pi/pow_1_grad/sub/yConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
}
 gradients/main/pi/pow_1_grad/subSubmain/pi/pow_1/y"gradients/main/pi/pow_1_grad/sub/y*
T0*
_output_shapes
: 
І
 gradients/main/pi/pow_1_grad/PowPowmain/pi/Tanh_1 gradients/main/pi/pow_1_grad/sub*
T0*'
_output_shapes
:         
Ъ
"gradients/main/pi/pow_1_grad/mul_1Mul gradients/main/pi/pow_1_grad/mul gradients/main/pi/pow_1_grad/Pow*'
_output_shapes
:         *
T0
┐
 gradients/main/pi/pow_1_grad/SumSum"gradients/main/pi/pow_1_grad/mul_12gradients/main/pi/pow_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
$gradients/main/pi/pow_1_grad/ReshapeReshape gradients/main/pi/pow_1_grad/Sum"gradients/main/pi/pow_1_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
k
&gradients/main/pi/pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
$gradients/main/pi/pow_1_grad/GreaterGreatermain/pi/Tanh_1&gradients/main/pi/pow_1_grad/Greater/y*
T0*'
_output_shapes
:         
z
,gradients/main/pi/pow_1_grad/ones_like/ShapeShapemain/pi/Tanh_1*
T0*
out_type0*
_output_shapes
:
q
,gradients/main/pi/pow_1_grad/ones_like/ConstConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╬
&gradients/main/pi/pow_1_grad/ones_likeFill,gradients/main/pi/pow_1_grad/ones_like/Shape,gradients/main/pi/pow_1_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:         
й
#gradients/main/pi/pow_1_grad/SelectSelect$gradients/main/pi/pow_1_grad/Greatermain/pi/Tanh_1&gradients/main/pi/pow_1_grad/ones_like*
T0*'
_output_shapes
:         
~
 gradients/main/pi/pow_1_grad/LogLog#gradients/main/pi/pow_1_grad/Select*
T0*'
_output_shapes
:         
v
'gradients/main/pi/pow_1_grad/zeros_like	ZerosLikemain/pi/Tanh_1*'
_output_shapes
:         *
T0
м
%gradients/main/pi/pow_1_grad/Select_1Select$gradients/main/pi/pow_1_grad/Greater gradients/main/pi/pow_1_grad/Log'gradients/main/pi/pow_1_grad/zeros_like*
T0*'
_output_shapes
:         
Б
"gradients/main/pi/pow_1_grad/mul_2Mul7gradients/main/pi/sub_2_grad/tuple/control_dependency_1main/pi/pow_1*
T0*'
_output_shapes
:         
д
"gradients/main/pi/pow_1_grad/mul_3Mul"gradients/main/pi/pow_1_grad/mul_2%gradients/main/pi/pow_1_grad/Select_1*
T0*'
_output_shapes
:         
├
"gradients/main/pi/pow_1_grad/Sum_1Sum"gradients/main/pi/pow_1_grad/mul_34gradients/main/pi/pow_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ф
&gradients/main/pi/pow_1_grad/Reshape_1Reshape"gradients/main/pi/pow_1_grad/Sum_1$gradients/main/pi/pow_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ё
-gradients/main/pi/pow_1_grad/tuple/group_depsNoOp%^gradients/main/pi/pow_1_grad/Reshape'^gradients/main/pi/pow_1_grad/Reshape_1
ѓ
5gradients/main/pi/pow_1_grad/tuple/control_dependencyIdentity$gradients/main/pi/pow_1_grad/Reshape.^gradients/main/pi/pow_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/pow_1_grad/Reshape*'
_output_shapes
:         
э
7gradients/main/pi/pow_1_grad/tuple/control_dependency_1Identity&gradients/main/pi/pow_1_grad/Reshape_1.^gradients/main/pi/pow_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/pow_1_grad/Reshape_1*
_output_shapes
: 
З
gradients/AddN_3AddN2gradients/main/mul_1_grad/tuple/control_dependency5gradients/main/pi/pow_1_grad/tuple/control_dependency*
N*'
_output_shapes
:         *
T0*4
_class*
(&loc:@gradients/main/mul_1_grad/Reshape
є
&gradients/main/pi/Tanh_1_grad/TanhGradTanhGradmain/pi/Tanh_1gradients/AddN_3*
T0*'
_output_shapes
:         
в
gradients/AddN_4AddN5gradients/main/pi/sub_1_grad/tuple/control_dependency&gradients/main/pi/Tanh_1_grad/TanhGrad*
T0*7
_class-
+)loc:@gradients/main/pi/sub_1_grad/Reshape*
N*'
_output_shapes
:         
y
"gradients/main/pi/add_2_grad/ShapeShapemain/pi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
q
$gradients/main/pi/add_2_grad/Shape_1Shapemain/pi/mul_1*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/add_2_grad/Shape$gradients/main/pi/add_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Г
 gradients/main/pi/add_2_grad/SumSumgradients/AddN_42gradients/main/pi/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
$gradients/main/pi/add_2_grad/ReshapeReshape gradients/main/pi/add_2_grad/Sum"gradients/main/pi/add_2_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
▒
"gradients/main/pi/add_2_grad/Sum_1Sumgradients/AddN_44gradients/main/pi/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╗
&gradients/main/pi/add_2_grad/Reshape_1Reshape"gradients/main/pi/add_2_grad/Sum_1$gradients/main/pi/add_2_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
Ё
-gradients/main/pi/add_2_grad/tuple/group_depsNoOp%^gradients/main/pi/add_2_grad/Reshape'^gradients/main/pi/add_2_grad/Reshape_1
ѓ
5gradients/main/pi/add_2_grad/tuple/control_dependencyIdentity$gradients/main/pi/add_2_grad/Reshape.^gradients/main/pi/add_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/add_2_grad/Reshape*'
_output_shapes
:         
ѕ
7gradients/main/pi/add_2_grad/tuple/control_dependency_1Identity&gradients/main/pi/add_2_grad/Reshape_1.^gradients/main/pi/add_2_grad/tuple/group_deps*'
_output_shapes
:         *
T0*9
_class/
-+loc:@gradients/main/pi/add_2_grad/Reshape_1
■
gradients/AddN_5AddN7gradients/main/pi/sub_1_grad/tuple/control_dependency_15gradients/main/pi/add_2_grad/tuple/control_dependency*
T0*9
_class/
-+loc:@gradients/main/pi/sub_1_grad/Reshape_1*
N*'
_output_shapes
:         
Ј
2gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes
:
Є
7gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_53^gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad
ё
?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_58^gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/sub_1_grad/Reshape_1*'
_output_shapes
:         
Д
Agradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*E
_class;
97loc:@gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad
w
"gradients/main/pi/mul_1_grad/ShapeShapemain/pi/random_normal*
T0*
out_type0*
_output_shapes
:
o
$gradients/main/pi/mul_1_grad/Shape_1Shapemain/pi/Exp*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/mul_1_grad/Shape$gradients/main/pi/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ъ
 gradients/main/pi/mul_1_grad/MulMul7gradients/main/pi/add_2_grad/tuple/control_dependency_1main/pi/Exp*
T0*'
_output_shapes
:         
й
 gradients/main/pi/mul_1_grad/SumSum gradients/main/pi/mul_1_grad/Mul2gradients/main/pi/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
$gradients/main/pi/mul_1_grad/ReshapeReshape gradients/main/pi/mul_1_grad/Sum"gradients/main/pi/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ф
"gradients/main/pi/mul_1_grad/Mul_1Mulmain/pi/random_normal7gradients/main/pi/add_2_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
├
"gradients/main/pi/mul_1_grad/Sum_1Sum"gradients/main/pi/mul_1_grad/Mul_14gradients/main/pi/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╗
&gradients/main/pi/mul_1_grad/Reshape_1Reshape"gradients/main/pi/mul_1_grad/Sum_1$gradients/main/pi/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ё
-gradients/main/pi/mul_1_grad/tuple/group_depsNoOp%^gradients/main/pi/mul_1_grad/Reshape'^gradients/main/pi/mul_1_grad/Reshape_1
ѓ
5gradients/main/pi/mul_1_grad/tuple/control_dependencyIdentity$gradients/main/pi/mul_1_grad/Reshape.^gradients/main/pi/mul_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/mul_1_grad/Reshape*'
_output_shapes
:         
ѕ
7gradients/main/pi/mul_1_grad/tuple/control_dependency_1Identity&gradients/main/pi/mul_1_grad/Reshape_1.^gradients/main/pi/mul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/mul_1_grad/Reshape_1*'
_output_shapes
:         
ь
,gradients/main/pi/dense_2/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencymain/pi/dense_2/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
▀
.gradients/main/pi/dense_2/MatMul_grad/MatMul_1MatMulmain/pi/dense_1/Relu?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( 
ъ
6gradients/main/pi/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_2/MatMul_grad/MatMul/^gradients/main/pi/dense_2/MatMul_grad/MatMul_1
Ц
>gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_2/MatMul_grad/MatMul7^gradients/main/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         ђ
б
@gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_2/MatMul_grad/MatMul_17^gradients/main/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	ђ
Ю
gradients/main/pi/Exp_grad/mulMul7gradients/main/pi/mul_1_grad/tuple/control_dependency_1main/pi/Exp*
T0*'
_output_shapes
:         
Ѕ
gradients/AddN_6AddN7gradients/main/pi/mul_2_grad/tuple/control_dependency_1 gradients/main/pi/Exp_1_grad/mulgradients/main/pi/Exp_grad/mul*
T0*9
_class/
-+loc:@gradients/main/pi/mul_2_grad/Reshape_1*
N*'
_output_shapes
:         
e
"gradients/main/pi/add_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
o
$gradients/main/pi/add_1_grad/Shape_1Shapemain/pi/mul*
T0*
out_type0*
_output_shapes
:
м
2gradients/main/pi/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/main/pi/add_1_grad/Shape$gradients/main/pi/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Г
 gradients/main/pi/add_1_grad/SumSumgradients/AddN_62gradients/main/pi/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ц
$gradients/main/pi/add_1_grad/ReshapeReshape gradients/main/pi/add_1_grad/Sum"gradients/main/pi/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
▒
"gradients/main/pi/add_1_grad/Sum_1Sumgradients/AddN_64gradients/main/pi/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╗
&gradients/main/pi/add_1_grad/Reshape_1Reshape"gradients/main/pi/add_1_grad/Sum_1$gradients/main/pi/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ё
-gradients/main/pi/add_1_grad/tuple/group_depsNoOp%^gradients/main/pi/add_1_grad/Reshape'^gradients/main/pi/add_1_grad/Reshape_1
ы
5gradients/main/pi/add_1_grad/tuple/control_dependencyIdentity$gradients/main/pi/add_1_grad/Reshape.^gradients/main/pi/add_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/add_1_grad/Reshape*
_output_shapes
: 
ѕ
7gradients/main/pi/add_1_grad/tuple/control_dependency_1Identity&gradients/main/pi/add_1_grad/Reshape_1.^gradients/main/pi/add_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/main/pi/add_1_grad/Reshape_1*'
_output_shapes
:         
c
 gradients/main/pi/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
m
"gradients/main/pi/mul_grad/Shape_1Shapemain/pi/add*
_output_shapes
:*
T0*
out_type0
╠
0gradients/main/pi/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/main/pi/mul_grad/Shape"gradients/main/pi/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ю
gradients/main/pi/mul_grad/MulMul7gradients/main/pi/add_1_grad/tuple/control_dependency_1main/pi/add*
T0*'
_output_shapes
:         
и
gradients/main/pi/mul_grad/SumSumgradients/main/pi/mul_grad/Mul0gradients/main/pi/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ъ
"gradients/main/pi/mul_grad/ReshapeReshapegradients/main/pi/mul_grad/Sum gradients/main/pi/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
А
 gradients/main/pi/mul_grad/Mul_1Mulmain/pi/mul/x7gradients/main/pi/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
й
 gradients/main/pi/mul_grad/Sum_1Sum gradients/main/pi/mul_grad/Mul_12gradients/main/pi/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
$gradients/main/pi/mul_grad/Reshape_1Reshape gradients/main/pi/mul_grad/Sum_1"gradients/main/pi/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         

+gradients/main/pi/mul_grad/tuple/group_depsNoOp#^gradients/main/pi/mul_grad/Reshape%^gradients/main/pi/mul_grad/Reshape_1
ж
3gradients/main/pi/mul_grad/tuple/control_dependencyIdentity"gradients/main/pi/mul_grad/Reshape,^gradients/main/pi/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/main/pi/mul_grad/Reshape*
_output_shapes
: 
ђ
5gradients/main/pi/mul_grad/tuple/control_dependency_1Identity$gradients/main/pi/mul_grad/Reshape_1,^gradients/main/pi/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/mul_grad/Reshape_1*'
_output_shapes
:         
t
 gradients/main/pi/add_grad/ShapeShapemain/pi/dense_3/Tanh*
T0*
out_type0*
_output_shapes
:
e
"gradients/main/pi/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╠
0gradients/main/pi/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/main/pi/add_grad/Shape"gradients/main/pi/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╬
gradients/main/pi/add_grad/SumSum5gradients/main/pi/mul_grad/tuple/control_dependency_10gradients/main/pi/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
»
"gradients/main/pi/add_grad/ReshapeReshapegradients/main/pi/add_grad/Sum gradients/main/pi/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
м
 gradients/main/pi/add_grad/Sum_1Sum5gradients/main/pi/mul_grad/tuple/control_dependency_12gradients/main/pi/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ц
$gradients/main/pi/add_grad/Reshape_1Reshape gradients/main/pi/add_grad/Sum_1"gradients/main/pi/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

+gradients/main/pi/add_grad/tuple/group_depsNoOp#^gradients/main/pi/add_grad/Reshape%^gradients/main/pi/add_grad/Reshape_1
Щ
3gradients/main/pi/add_grad/tuple/control_dependencyIdentity"gradients/main/pi/add_grad/Reshape,^gradients/main/pi/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/main/pi/add_grad/Reshape*'
_output_shapes
:         
№
5gradients/main/pi/add_grad/tuple/control_dependency_1Identity$gradients/main/pi/add_grad/Reshape_1,^gradients/main/pi/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/main/pi/add_grad/Reshape_1*
_output_shapes
: 
х
,gradients/main/pi/dense_3/Tanh_grad/TanhGradTanhGradmain/pi/dense_3/Tanh3gradients/main/pi/add_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
Ф
2gradients/main/pi/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/pi/dense_3/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:
Б
7gradients/main/pi/dense_3/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/pi/dense_3/BiasAdd_grad/BiasAddGrad-^gradients/main/pi/dense_3/Tanh_grad/TanhGrad
д
?gradients/main/pi/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_3/Tanh_grad/TanhGrad8^gradients/main/pi/dense_3/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_3/Tanh_grad/TanhGrad*'
_output_shapes
:         
Д
Agradients/main/pi/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_3/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_3/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/main/pi/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ь
,gradients/main/pi/dense_3/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_3/BiasAdd_grad/tuple/control_dependencymain/pi/dense_3/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
▀
.gradients/main/pi/dense_3/MatMul_grad/MatMul_1MatMulmain/pi/dense_1/Relu?gradients/main/pi/dense_3/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( *
T0
ъ
6gradients/main/pi/dense_3/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_3/MatMul_grad/MatMul/^gradients/main/pi/dense_3/MatMul_grad/MatMul_1
Ц
>gradients/main/pi/dense_3/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_3/MatMul_grad/MatMul7^gradients/main/pi/dense_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_3/MatMul_grad/MatMul*(
_output_shapes
:         ђ
б
@gradients/main/pi/dense_3/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_3/MatMul_grad/MatMul_17^gradients/main/pi/dense_3/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/pi/dense_3/MatMul_grad/MatMul_1*
_output_shapes
:	ђ
Ћ
gradients/AddN_7AddN>gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency>gradients/main/pi/dense_3/MatMul_grad/tuple/control_dependency*
T0*?
_class5
31loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul*
N*(
_output_shapes
:         ђ
Њ
,gradients/main/pi/dense_1/Relu_grad/ReluGradReluGradgradients/AddN_7main/pi/dense_1/Relu*(
_output_shapes
:         ђ*
T0
г
2gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/pi/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:ђ
Б
7gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad-^gradients/main/pi/dense_1/Relu_grad/ReluGrad
Д
?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_1/Relu_grad/ReluGrad8^gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:         ђ
е
Agradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*E
_class;
97loc:@gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad
ь
,gradients/main/pi/dense_1/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencymain/pi/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
я
.gradients/main/pi/dense_1/MatMul_grad/MatMul_1MatMulmain/pi/dense/Relu?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
ђђ*
transpose_a(
ъ
6gradients/main/pi/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_1/MatMul_grad/MatMul/^gradients/main/pi/dense_1/MatMul_grad/MatMul_1
Ц
>gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_1/MatMul_grad/MatMul7^gradients/main/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Б
@gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_1/MatMul_grad/MatMul_17^gradients/main/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/pi/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
й
*gradients/main/pi/dense/Relu_grad/ReluGradReluGrad>gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependencymain/pi/dense/Relu*
T0*(
_output_shapes
:         ђ
е
0gradients/main/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/main/pi/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:ђ*
T0
Ю
5gradients/main/pi/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad+^gradients/main/pi/dense/Relu_grad/ReluGrad
Ъ
=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/main/pi/dense/Relu_grad/ReluGrad6^gradients/main/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/main/pi/dense/Relu_grad/ReluGrad*(
_output_shapes
:         ђ
а
?gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad6^gradients/main/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Т
*gradients/main/pi/dense/MatMul_grad/MatMulMatMul=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependencymain/pi/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:         I*
transpose_a( 
м
,gradients/main/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	Iђ*
transpose_a(*
transpose_b( 
ў
4gradients/main/pi/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/main/pi/dense/MatMul_grad/MatMul-^gradients/main/pi/dense/MatMul_grad/MatMul_1
ю
<gradients/main/pi/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/main/pi/dense/MatMul_grad/MatMul5^gradients/main/pi/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:         I*
T0*=
_class3
1/loc:@gradients/main/pi/dense/MatMul_grad/MatMul
џ
>gradients/main/pi/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/main/pi/dense/MatMul_grad/MatMul_15^gradients/main/pi/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	Iђ
Ё
beta1_power/initial_valueConst*
valueB
 *fff?*%
_class
loc:@main/pi/dense/bias*
dtype0*
_output_shapes
: 
ќ
beta1_power
VariableV2*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
х
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
q
beta1_power/readIdentitybeta1_power*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
Ё
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wЙ?*%
_class
loc:@main/pi/dense/bias
ќ
beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@main/pi/dense/bias*
	container 
х
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
q
beta2_power/readIdentitybeta2_power*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
х
;main/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@main/pi/dense/kernel*
valueB"I      *
dtype0*
_output_shapes
:
Ъ
1main/pi/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@main/pi/dense/kernel*
valueB
 *    
ѕ
+main/pi/dense/kernel/Adam/Initializer/zerosFill;main/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1main/pi/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	Iђ*
T0*'
_class
loc:@main/pi/dense/kernel*

index_type0
И
main/pi/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	Iђ*
shared_name *'
_class
loc:@main/pi/dense/kernel*
	container *
shape:	Iђ
Ь
 main/pi/dense/kernel/Adam/AssignAssignmain/pi/dense/kernel/Adam+main/pi/dense/kernel/Adam/Initializer/zeros*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ*
use_locking(
ў
main/pi/dense/kernel/Adam/readIdentitymain/pi/dense/kernel/Adam*
_output_shapes
:	Iђ*
T0*'
_class
loc:@main/pi/dense/kernel
и
=main/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@main/pi/dense/kernel*
valueB"I      *
dtype0*
_output_shapes
:
А
3main/pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@main/pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
-main/pi/dense/kernel/Adam_1/Initializer/zerosFill=main/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3main/pi/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*'
_class
loc:@main/pi/dense/kernel*

index_type0*
_output_shapes
:	Iђ
║
main/pi/dense/kernel/Adam_1
VariableV2*
shared_name *'
_class
loc:@main/pi/dense/kernel*
	container *
shape:	Iђ*
dtype0*
_output_shapes
:	Iђ
З
"main/pi/dense/kernel/Adam_1/AssignAssignmain/pi/dense/kernel/Adam_1-main/pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
ю
 main/pi/dense/kernel/Adam_1/readIdentitymain/pi/dense/kernel/Adam_1*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	Iђ
Ф
9main/pi/dense/bias/Adam/Initializer/zeros/shape_as_tensorConst*%
_class
loc:@main/pi/dense/bias*
valueB:ђ*
dtype0*
_output_shapes
:
Џ
/main/pi/dense/bias/Adam/Initializer/zeros/ConstConst*%
_class
loc:@main/pi/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
)main/pi/dense/bias/Adam/Initializer/zerosFill9main/pi/dense/bias/Adam/Initializer/zeros/shape_as_tensor/main/pi/dense/bias/Adam/Initializer/zeros/Const*
T0*%
_class
loc:@main/pi/dense/bias*

index_type0*
_output_shapes	
:ђ
г
main/pi/dense/bias/Adam
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container 
Р
main/pi/dense/bias/Adam/AssignAssignmain/pi/dense/bias/Adam)main/pi/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ
ј
main/pi/dense/bias/Adam/readIdentitymain/pi/dense/bias/Adam*
_output_shapes	
:ђ*
T0*%
_class
loc:@main/pi/dense/bias
Г
;main/pi/dense/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*%
_class
loc:@main/pi/dense/bias*
valueB:ђ*
dtype0*
_output_shapes
:
Ю
1main/pi/dense/bias/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
valueB
 *    
ѓ
+main/pi/dense/bias/Adam_1/Initializer/zerosFill;main/pi/dense/bias/Adam_1/Initializer/zeros/shape_as_tensor1main/pi/dense/bias/Adam_1/Initializer/zeros/Const*
T0*%
_class
loc:@main/pi/dense/bias*

index_type0*
_output_shapes	
:ђ
«
main/pi/dense/bias/Adam_1
VariableV2*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
У
 main/pi/dense/bias/Adam_1/AssignAssignmain/pi/dense/bias/Adam_1+main/pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ
њ
main/pi/dense/bias/Adam_1/readIdentitymain/pi/dense/bias/Adam_1*
_output_shapes	
:ђ*
T0*%
_class
loc:@main/pi/dense/bias
╣
=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Б
3main/pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
-main/pi/dense_1/kernel/Adam/Initializer/zerosFill=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/pi/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@main/pi/dense_1/kernel*

index_type0* 
_output_shapes
:
ђђ
Й
main/pi/dense_1/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *)
_class
loc:@main/pi/dense_1/kernel*
	container *
shape:
ђђ
э
"main/pi/dense_1/kernel/Adam/AssignAssignmain/pi/dense_1/kernel/Adam-main/pi/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Ъ
 main/pi/dense_1/kernel/Adam/readIdentitymain/pi/dense_1/kernel/Adam* 
_output_shapes
:
ђђ*
T0*)
_class
loc:@main/pi/dense_1/kernel
╗
?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
/main/pi/dense_1/kernel/Adam_1/Initializer/zerosFill?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@main/pi/dense_1/kernel*

index_type0* 
_output_shapes
:
ђђ
└
main/pi/dense_1/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@main/pi/dense_1/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ
§
$main/pi/dense_1/kernel/Adam_1/AssignAssignmain/pi/dense_1/kernel/Adam_1/main/pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Б
"main/pi/dense_1/kernel/Adam_1/readIdentitymain/pi/dense_1/kernel/Adam_1*
T0*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
ђђ
Б
+main/pi/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
░
main/pi/dense_1/bias/Adam
VariableV2*
shared_name *'
_class
loc:@main/pi/dense_1/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
Ж
 main/pi/dense_1/bias/Adam/AssignAssignmain/pi/dense_1/bias/Adam+main/pi/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
ћ
main/pi/dense_1/bias/Adam/readIdentitymain/pi/dense_1/bias/Adam*
T0*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:ђ
Ц
-main/pi/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/pi/dense_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
▓
main/pi/dense_1/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@main/pi/dense_1/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
­
"main/pi/dense_1/bias/Adam_1/AssignAssignmain/pi/dense_1/bias/Adam_1-main/pi/dense_1/bias/Adam_1/Initializer/zeros*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
ў
 main/pi/dense_1/bias/Adam_1/readIdentitymain/pi/dense_1/bias/Adam_1*
_output_shapes	
:ђ*
T0*'
_class
loc:@main/pi/dense_1/bias
╣
=main/pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*)
_class
loc:@main/pi/dense_2/kernel*
valueB"      
Б
3main/pi/dense_2/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@main/pi/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
љ
-main/pi/dense_2/kernel/Adam/Initializer/zerosFill=main/pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor3main/pi/dense_2/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/pi/dense_2/kernel*

index_type0
╝
main/pi/dense_2/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
Ш
"main/pi/dense_2/kernel/Adam/AssignAssignmain/pi/dense_2/kernel/Adam-main/pi/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel
ъ
 main/pi/dense_2/kernel/Adam/readIdentitymain/pi/dense_2/kernel/Adam*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/pi/dense_2/kernel
╗
?main/pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
5main/pi/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@main/pi/dense_2/kernel*
valueB
 *    
ќ
/main/pi/dense_2/kernel/Adam_1/Initializer/zerosFill?main/pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/pi/dense_2/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@main/pi/dense_2/kernel*

index_type0*
_output_shapes
:	ђ
Й
main/pi/dense_2/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
Ч
$main/pi/dense_2/kernel/Adam_1/AssignAssignmain/pi/dense_2/kernel/Adam_1/main/pi/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel
б
"main/pi/dense_2/kernel/Adam_1/readIdentitymain/pi/dense_2/kernel/Adam_1*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/pi/dense_2/kernel
А
+main/pi/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
«
main/pi/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container *
shape:
ж
 main/pi/dense_2/bias/Adam/AssignAssignmain/pi/dense_2/bias/Adam+main/pi/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias
Њ
main/pi/dense_2/bias/Adam/readIdentitymain/pi/dense_2/bias/Adam*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_2/bias
Б
-main/pi/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
░
main/pi/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container *
shape:
№
"main/pi/dense_2/bias/Adam_1/AssignAssignmain/pi/dense_2/bias/Adam_1-main/pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
Ќ
 main/pi/dense_2/bias/Adam_1/readIdentitymain/pi/dense_2/bias/Adam_1*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_2/bias
╣
=main/pi/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Б
3main/pi/dense_3/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@main/pi/dense_3/kernel*
valueB
 *    
љ
-main/pi/dense_3/kernel/Adam/Initializer/zerosFill=main/pi/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensor3main/pi/dense_3/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/pi/dense_3/kernel*

index_type0
╝
main/pi/dense_3/kernel/Adam
VariableV2*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name *)
_class
loc:@main/pi/dense_3/kernel
Ш
"main/pi/dense_3/kernel/Adam/AssignAssignmain/pi/dense_3/kernel/Adam-main/pi/dense_3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ
ъ
 main/pi/dense_3/kernel/Adam/readIdentitymain/pi/dense_3/kernel/Adam*
T0*)
_class
loc:@main/pi/dense_3/kernel*
_output_shapes
:	ђ
╗
?main/pi/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
5main/pi/dense_3/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@main/pi/dense_3/kernel*
valueB
 *    
ќ
/main/pi/dense_3/kernel/Adam_1/Initializer/zerosFill?main/pi/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/pi/dense_3/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@main/pi/dense_3/kernel*

index_type0*
_output_shapes
:	ђ
Й
main/pi/dense_3/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@main/pi/dense_3/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
Ч
$main/pi/dense_3/kernel/Adam_1/AssignAssignmain/pi/dense_3/kernel/Adam_1/main/pi/dense_3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ
б
"main/pi/dense_3/kernel/Adam_1/readIdentitymain/pi/dense_3/kernel/Adam_1*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/pi/dense_3/kernel
А
+main/pi/dense_3/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_3/bias*
valueB*    *
dtype0*
_output_shapes
:
«
main/pi/dense_3/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_3/bias*
	container *
shape:
ж
 main/pi/dense_3/bias/Adam/AssignAssignmain/pi/dense_3/bias/Adam+main/pi/dense_3/bias/Adam/Initializer/zeros*
T0*'
_class
loc:@main/pi/dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Њ
main/pi/dense_3/bias/Adam/readIdentitymain/pi/dense_3/bias/Adam*
T0*'
_class
loc:@main/pi/dense_3/bias*
_output_shapes
:
Б
-main/pi/dense_3/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*'
_class
loc:@main/pi/dense_3/bias*
valueB*    
░
main/pi/dense_3/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@main/pi/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes
:
№
"main/pi/dense_3/bias/Adam_1/AssignAssignmain/pi/dense_3/bias/Adam_1-main/pi/dense_3/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_3/bias
Ќ
 main/pi/dense_3/bias/Adam_1/readIdentitymain/pi/dense_3/bias/Adam_1*
T0*'
_class
loc:@main/pi/dense_3/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
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
 *wЙ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
Ю
*Adam/update_main/pi/dense/kernel/ApplyAdam	ApplyAdammain/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/pi/dense/MatMul_grad/tuple/control_dependency_1*
T0*'
_class
loc:@main/pi/dense/kernel*
use_nesterov( *
_output_shapes
:	Iђ*
use_locking( 
љ
(Adam/update_main/pi/dense/bias/ApplyAdam	ApplyAdammain/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@main/pi/dense/bias*
use_nesterov( *
_output_shapes	
:ђ
ф
,Adam/update_main/pi/dense_1/kernel/ApplyAdam	ApplyAdammain/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@main/pi/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
ђђ
ю
*Adam/update_main/pi/dense_1/bias/ApplyAdam	ApplyAdammain/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@main/pi/dense_1/bias*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( 
Е
,Adam/update_main/pi/dense_2/kernel/ApplyAdam	ApplyAdammain/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	ђ*
use_locking( *
T0*)
_class
loc:@main/pi/dense_2/kernel
Џ
*Adam/update_main/pi/dense_2/bias/ApplyAdam	ApplyAdammain/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/pi/dense_2/bias*
use_nesterov( *
_output_shapes
:
Е
,Adam/update_main/pi/dense_3/kernel/ApplyAdam	ApplyAdammain/pi/dense_3/kernelmain/pi/dense_3/kernel/Adammain/pi/dense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@main/pi/dense_3/kernel*
use_nesterov( *
_output_shapes
:	ђ
Џ
*Adam/update_main/pi/dense_3/bias/ApplyAdam	ApplyAdammain/pi/dense_3/biasmain/pi/dense_3/bias/Adammain/pi/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/pi/dense_3/bias*
use_nesterov( *
_output_shapes
:
р
Adam/mulMulbeta1_power/read
Adam/beta1)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam+^Adam/update_main/pi/dense_3/bias/ApplyAdam-^Adam/update_main/pi/dense_3/kernel/ApplyAdam*
_output_shapes
: *
T0*%
_class
loc:@main/pi/dense/bias
Ю
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*%
_class
loc:@main/pi/dense/bias
с

Adam/mul_1Mulbeta2_power/read
Adam/beta2)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam+^Adam/update_main/pi/dense_3/bias/ApplyAdam-^Adam/update_main/pi/dense_3/kernel/ApplyAdam*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
А
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
ќ
AdamNoOp^Adam/Assign^Adam/Assign_1)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam+^Adam/update_main/pi/dense_3/bias/ApplyAdam-^Adam/update_main/pi/dense_3/kernel/ApplyAdam
[
gradients_1/ShapeConst^Adam*
dtype0*
_output_shapes
: *
valueB 
a
gradients_1/grad_ys_0Const^Adam*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
I
'gradients_1/add_2_grad/tuple/group_depsNoOp^Adam^gradients_1/Fill
й
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/Fill(^gradients_1/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*#
_class
loc:@gradients_1/Fill
┐
1gradients_1/add_2_grad/tuple/control_dependency_1Identitygradients_1/Fill(^gradients_1/add_2_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill*
_output_shapes
: 
h
'gradients_1/add_1_grad/tuple/group_depsNoOp^Adam0^gradients_1/add_2_grad/tuple/control_dependency
▄
/gradients_1/add_1_grad/tuple/control_dependencyIdentity/gradients_1/add_2_grad/tuple/control_dependency(^gradients_1/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*#
_class
loc:@gradients_1/Fill
я
1gradients_1/add_1_grad/tuple/control_dependency_1Identity/gradients_1/add_2_grad/tuple/control_dependency(^gradients_1/add_1_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill*
_output_shapes
: 
}
gradients_1/mul_6_grad/MulMul1gradients_1/add_2_grad/tuple/control_dependency_1Mean_3*
_output_shapes
: *
T0
ђ
gradients_1/mul_6_grad/Mul_1Mul1gradients_1/add_2_grad/tuple/control_dependency_1mul_6/x*
T0*
_output_shapes
: 
r
'gradients_1/mul_6_grad/tuple/group_depsNoOp^Adam^gradients_1/mul_6_grad/Mul^gradients_1/mul_6_grad/Mul_1
Л
/gradients_1/mul_6_grad/tuple/control_dependencyIdentitygradients_1/mul_6_grad/Mul(^gradients_1/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_6_grad/Mul*
_output_shapes
: 
О
1gradients_1/mul_6_grad/tuple/control_dependency_1Identitygradients_1/mul_6_grad/Mul_1(^gradients_1/mul_6_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients_1/mul_6_grad/Mul_1
{
gradients_1/mul_4_grad/MulMul/gradients_1/add_1_grad/tuple/control_dependencyMean_1*
_output_shapes
: *
T0
~
gradients_1/mul_4_grad/Mul_1Mul/gradients_1/add_1_grad/tuple/control_dependencymul_4/x*
T0*
_output_shapes
: 
r
'gradients_1/mul_4_grad/tuple/group_depsNoOp^Adam^gradients_1/mul_4_grad/Mul^gradients_1/mul_4_grad/Mul_1
Л
/gradients_1/mul_4_grad/tuple/control_dependencyIdentitygradients_1/mul_4_grad/Mul(^gradients_1/mul_4_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients_1/mul_4_grad/Mul
О
1gradients_1/mul_4_grad/tuple/control_dependency_1Identitygradients_1/mul_4_grad/Mul_1(^gradients_1/mul_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_4_grad/Mul_1*
_output_shapes
: 
}
gradients_1/mul_5_grad/MulMul1gradients_1/add_1_grad/tuple/control_dependency_1Mean_2*
_output_shapes
: *
T0
ђ
gradients_1/mul_5_grad/Mul_1Mul1gradients_1/add_1_grad/tuple/control_dependency_1mul_5/x*
T0*
_output_shapes
: 
r
'gradients_1/mul_5_grad/tuple/group_depsNoOp^Adam^gradients_1/mul_5_grad/Mul^gradients_1/mul_5_grad/Mul_1
Л
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Mul(^gradients_1/mul_5_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients_1/mul_5_grad/Mul
О
1gradients_1/mul_5_grad/tuple/control_dependency_1Identitygradients_1/mul_5_grad/Mul_1(^gradients_1/mul_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_5_grad/Mul_1*
_output_shapes
: 
v
%gradients_1/Mean_3_grad/Reshape/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
и
gradients_1/Mean_3_grad/ReshapeReshape1gradients_1/mul_6_grad/tuple/control_dependency_1%gradients_1/Mean_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
i
gradients_1/Mean_3_grad/ShapeShapepow_2^Adam*
T0*
out_type0*
_output_shapes
:
ц
gradients_1/Mean_3_grad/TileTilegradients_1/Mean_3_grad/Reshapegradients_1/Mean_3_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
k
gradients_1/Mean_3_grad/Shape_1Shapepow_2^Adam*
T0*
out_type0*
_output_shapes
:
i
gradients_1/Mean_3_grad/Shape_2Const^Adam*
valueB *
dtype0*
_output_shapes
: 
n
gradients_1/Mean_3_grad/ConstConst^Adam*
valueB: *
dtype0*
_output_shapes
:
б
gradients_1/Mean_3_grad/ProdProdgradients_1/Mean_3_grad/Shape_1gradients_1/Mean_3_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
p
gradients_1/Mean_3_grad/Const_1Const^Adam*
valueB: *
dtype0*
_output_shapes
:
д
gradients_1/Mean_3_grad/Prod_1Prodgradients_1/Mean_3_grad/Shape_2gradients_1/Mean_3_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
!gradients_1/Mean_3_grad/Maximum/yConst^Adam*
value	B :*
dtype0*
_output_shapes
: 
ј
gradients_1/Mean_3_grad/MaximumMaximumgradients_1/Mean_3_grad/Prod_1!gradients_1/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
ї
 gradients_1/Mean_3_grad/floordivFloorDivgradients_1/Mean_3_grad/Prodgradients_1/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
є
gradients_1/Mean_3_grad/CastCast gradients_1/Mean_3_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
ћ
gradients_1/Mean_3_grad/truedivRealDivgradients_1/Mean_3_grad/Tilegradients_1/Mean_3_grad/Cast*#
_output_shapes
:         *
T0
v
%gradients_1/Mean_1_grad/Reshape/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
и
gradients_1/Mean_1_grad/ReshapeReshape1gradients_1/mul_4_grad/tuple/control_dependency_1%gradients_1/Mean_1_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
g
gradients_1/Mean_1_grad/ShapeShapepow^Adam*
T0*
out_type0*
_output_shapes
:
ц
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
i
gradients_1/Mean_1_grad/Shape_1Shapepow^Adam*
_output_shapes
:*
T0*
out_type0
i
gradients_1/Mean_1_grad/Shape_2Const^Adam*
dtype0*
_output_shapes
: *
valueB 
n
gradients_1/Mean_1_grad/ConstConst^Adam*
valueB: *
dtype0*
_output_shapes
:
б
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
p
gradients_1/Mean_1_grad/Const_1Const^Adam*
dtype0*
_output_shapes
:*
valueB: 
д
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
!gradients_1/Mean_1_grad/Maximum/yConst^Adam*
dtype0*
_output_shapes
: *
value	B :
ј
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
ї
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
є
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
ћ
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:         
v
%gradients_1/Mean_2_grad/Reshape/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
и
gradients_1/Mean_2_grad/ReshapeReshape1gradients_1/mul_5_grad/tuple/control_dependency_1%gradients_1/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
i
gradients_1/Mean_2_grad/ShapeShapepow_1^Adam*
T0*
out_type0*
_output_shapes
:
ц
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*#
_output_shapes
:         *

Tmultiples0*
T0
k
gradients_1/Mean_2_grad/Shape_1Shapepow_1^Adam*
T0*
out_type0*
_output_shapes
:
i
gradients_1/Mean_2_grad/Shape_2Const^Adam*
valueB *
dtype0*
_output_shapes
: 
n
gradients_1/Mean_2_grad/ConstConst^Adam*
dtype0*
_output_shapes
:*
valueB: 
б
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
gradients_1/Mean_2_grad/Const_1Const^Adam*
valueB: *
dtype0*
_output_shapes
:
д
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
!gradients_1/Mean_2_grad/Maximum/yConst^Adam*
value	B :*
dtype0*
_output_shapes
: 
ј
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
ї
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
є
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
ћ
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*
T0*#
_output_shapes
:         
h
gradients_1/pow_2_grad/ShapeShapesub_5^Adam*
T0*
out_type0*
_output_shapes
:
h
gradients_1/pow_2_grad/Shape_1Const^Adam*
valueB *
dtype0*
_output_shapes
: 
└
,gradients_1/pow_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_2_grad/Shapegradients_1/pow_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
y
gradients_1/pow_2_grad/mulMulgradients_1/Mean_3_grad/truedivpow_2/y*
T0*#
_output_shapes
:         
h
gradients_1/pow_2_grad/sub/yConst^Adam*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
i
gradients_1/pow_2_grad/subSubpow_2/ygradients_1/pow_2_grad/sub/y*
T0*
_output_shapes
: 
r
gradients_1/pow_2_grad/PowPowsub_5gradients_1/pow_2_grad/sub*
T0*#
_output_shapes
:         
Ѕ
gradients_1/pow_2_grad/mul_1Mulgradients_1/pow_2_grad/mulgradients_1/pow_2_grad/Pow*#
_output_shapes
:         *
T0
Г
gradients_1/pow_2_grad/SumSumgradients_1/pow_2_grad/mul_1,gradients_1/pow_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ъ
gradients_1/pow_2_grad/ReshapeReshapegradients_1/pow_2_grad/Sumgradients_1/pow_2_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
l
 gradients_1/pow_2_grad/Greater/yConst^Adam*
valueB
 *    *
dtype0*
_output_shapes
: 
ђ
gradients_1/pow_2_grad/GreaterGreatersub_5 gradients_1/pow_2_grad/Greater/y*
T0*#
_output_shapes
:         
r
&gradients_1/pow_2_grad/ones_like/ShapeShapesub_5^Adam*
_output_shapes
:*
T0*
out_type0
r
&gradients_1/pow_2_grad/ones_like/ConstConst^Adam*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
И
 gradients_1/pow_2_grad/ones_likeFill&gradients_1/pow_2_grad/ones_like/Shape&gradients_1/pow_2_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:         
ъ
gradients_1/pow_2_grad/SelectSelectgradients_1/pow_2_grad/Greatersub_5 gradients_1/pow_2_grad/ones_like*
T0*#
_output_shapes
:         
n
gradients_1/pow_2_grad/LogLoggradients_1/pow_2_grad/Select*
T0*#
_output_shapes
:         
j
!gradients_1/pow_2_grad/zeros_like	ZerosLikesub_5^Adam*
T0*#
_output_shapes
:         
Х
gradients_1/pow_2_grad/Select_1Selectgradients_1/pow_2_grad/Greatergradients_1/pow_2_grad/Log!gradients_1/pow_2_grad/zeros_like*
T0*#
_output_shapes
:         
y
gradients_1/pow_2_grad/mul_2Mulgradients_1/Mean_3_grad/truedivpow_2*#
_output_shapes
:         *
T0
љ
gradients_1/pow_2_grad/mul_3Mulgradients_1/pow_2_grad/mul_2gradients_1/pow_2_grad/Select_1*
T0*#
_output_shapes
:         
▒
gradients_1/pow_2_grad/Sum_1Sumgradients_1/pow_2_grad/mul_3.gradients_1/pow_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ў
 gradients_1/pow_2_grad/Reshape_1Reshapegradients_1/pow_2_grad/Sum_1gradients_1/pow_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
z
'gradients_1/pow_2_grad/tuple/group_depsNoOp^Adam^gradients_1/pow_2_grad/Reshape!^gradients_1/pow_2_grad/Reshape_1
Т
/gradients_1/pow_2_grad/tuple/control_dependencyIdentitygradients_1/pow_2_grad/Reshape(^gradients_1/pow_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/pow_2_grad/Reshape*#
_output_shapes
:         
▀
1gradients_1/pow_2_grad/tuple/control_dependency_1Identity gradients_1/pow_2_grad/Reshape_1(^gradients_1/pow_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/pow_2_grad/Reshape_1*
_output_shapes
: 
f
gradients_1/pow_grad/ShapeShapesub_3^Adam*
_output_shapes
:*
T0*
out_type0
f
gradients_1/pow_grad/Shape_1Const^Adam*
valueB *
dtype0*
_output_shapes
: 
║
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*
T0*#
_output_shapes
:         
f
gradients_1/pow_grad/sub/yConst^Adam*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
T0*
_output_shapes
: 
n
gradients_1/pow_grad/PowPowsub_3gradients_1/pow_grad/sub*
T0*#
_output_shapes
:         
Ѓ
gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*#
_output_shapes
:         *
T0
Д
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ў
gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
j
gradients_1/pow_grad/Greater/yConst^Adam*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients_1/pow_grad/GreaterGreatersub_3gradients_1/pow_grad/Greater/y*
T0*#
_output_shapes
:         
p
$gradients_1/pow_grad/ones_like/ShapeShapesub_3^Adam*
T0*
out_type0*
_output_shapes
:
p
$gradients_1/pow_grad/ones_like/ConstConst^Adam*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
▓
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:         
ў
gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_3gradients_1/pow_grad/ones_like*
T0*#
_output_shapes
:         
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:         
h
gradients_1/pow_grad/zeros_like	ZerosLikesub_3^Adam*
T0*#
_output_shapes
:         
«
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*
T0*#
_output_shapes
:         
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*
T0*#
_output_shapes
:         
і
gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*
T0*#
_output_shapes
:         
Ф
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
њ
gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
t
%gradients_1/pow_grad/tuple/group_depsNoOp^Adam^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
я
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*#
_output_shapes
:         
О
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1*
_output_shapes
: 
h
gradients_1/pow_1_grad/ShapeShapesub_4^Adam*
T0*
out_type0*
_output_shapes
:
h
gradients_1/pow_1_grad/Shape_1Const^Adam*
valueB *
dtype0*
_output_shapes
: 
└
,gradients_1/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_1_grad/Shapegradients_1/pow_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
y
gradients_1/pow_1_grad/mulMulgradients_1/Mean_2_grad/truedivpow_1/y*
T0*#
_output_shapes
:         
h
gradients_1/pow_1_grad/sub/yConst^Adam*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
i
gradients_1/pow_1_grad/subSubpow_1/ygradients_1/pow_1_grad/sub/y*
T0*
_output_shapes
: 
r
gradients_1/pow_1_grad/PowPowsub_4gradients_1/pow_1_grad/sub*
T0*#
_output_shapes
:         
Ѕ
gradients_1/pow_1_grad/mul_1Mulgradients_1/pow_1_grad/mulgradients_1/pow_1_grad/Pow*
T0*#
_output_shapes
:         
Г
gradients_1/pow_1_grad/SumSumgradients_1/pow_1_grad/mul_1,gradients_1/pow_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
gradients_1/pow_1_grad/ReshapeReshapegradients_1/pow_1_grad/Sumgradients_1/pow_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
l
 gradients_1/pow_1_grad/Greater/yConst^Adam*
dtype0*
_output_shapes
: *
valueB
 *    
ђ
gradients_1/pow_1_grad/GreaterGreatersub_4 gradients_1/pow_1_grad/Greater/y*#
_output_shapes
:         *
T0
r
&gradients_1/pow_1_grad/ones_like/ShapeShapesub_4^Adam*
T0*
out_type0*
_output_shapes
:
r
&gradients_1/pow_1_grad/ones_like/ConstConst^Adam*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
И
 gradients_1/pow_1_grad/ones_likeFill&gradients_1/pow_1_grad/ones_like/Shape&gradients_1/pow_1_grad/ones_like/Const*#
_output_shapes
:         *
T0*

index_type0
ъ
gradients_1/pow_1_grad/SelectSelectgradients_1/pow_1_grad/Greatersub_4 gradients_1/pow_1_grad/ones_like*#
_output_shapes
:         *
T0
n
gradients_1/pow_1_grad/LogLoggradients_1/pow_1_grad/Select*
T0*#
_output_shapes
:         
j
!gradients_1/pow_1_grad/zeros_like	ZerosLikesub_4^Adam*
T0*#
_output_shapes
:         
Х
gradients_1/pow_1_grad/Select_1Selectgradients_1/pow_1_grad/Greatergradients_1/pow_1_grad/Log!gradients_1/pow_1_grad/zeros_like*
T0*#
_output_shapes
:         
y
gradients_1/pow_1_grad/mul_2Mulgradients_1/Mean_2_grad/truedivpow_1*#
_output_shapes
:         *
T0
љ
gradients_1/pow_1_grad/mul_3Mulgradients_1/pow_1_grad/mul_2gradients_1/pow_1_grad/Select_1*#
_output_shapes
:         *
T0
▒
gradients_1/pow_1_grad/Sum_1Sumgradients_1/pow_1_grad/mul_3.gradients_1/pow_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ў
 gradients_1/pow_1_grad/Reshape_1Reshapegradients_1/pow_1_grad/Sum_1gradients_1/pow_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
z
'gradients_1/pow_1_grad/tuple/group_depsNoOp^Adam^gradients_1/pow_1_grad/Reshape!^gradients_1/pow_1_grad/Reshape_1
Т
/gradients_1/pow_1_grad/tuple/control_dependencyIdentitygradients_1/pow_1_grad/Reshape(^gradients_1/pow_1_grad/tuple/group_deps*#
_output_shapes
:         *
T0*1
_class'
%#loc:@gradients_1/pow_1_grad/Reshape
▀
1gradients_1/pow_1_grad/tuple/control_dependency_1Identity gradients_1/pow_1_grad/Reshape_1(^gradients_1/pow_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/pow_1_grad/Reshape_1*
_output_shapes
: 
q
gradients_1/sub_5_grad/ShapeShapeStopGradient_1^Adam*
T0*
out_type0*
_output_shapes
:
s
gradients_1/sub_5_grad/Shape_1Shapemain/v/Squeeze^Adam*
T0*
out_type0*
_output_shapes
:
└
,gradients_1/sub_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_5_grad/Shapegradients_1/sub_5_grad/Shape_1*
T0*2
_output_shapes 
:         :         
└
gradients_1/sub_5_grad/SumSum/gradients_1/pow_2_grad/tuple/control_dependency,gradients_1/sub_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
gradients_1/sub_5_grad/ReshapeReshapegradients_1/sub_5_grad/Sumgradients_1/sub_5_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
─
gradients_1/sub_5_grad/Sum_1Sum/gradients_1/pow_2_grad/tuple/control_dependency.gradients_1/sub_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_1/sub_5_grad/NegNeggradients_1/sub_5_grad/Sum_1*
T0*
_output_shapes
:
Б
 gradients_1/sub_5_grad/Reshape_1Reshapegradients_1/sub_5_grad/Neggradients_1/sub_5_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
z
'gradients_1/sub_5_grad/tuple/group_depsNoOp^Adam^gradients_1/sub_5_grad/Reshape!^gradients_1/sub_5_grad/Reshape_1
Т
/gradients_1/sub_5_grad/tuple/control_dependencyIdentitygradients_1/sub_5_grad/Reshape(^gradients_1/sub_5_grad/tuple/group_deps*#
_output_shapes
:         *
T0*1
_class'
%#loc:@gradients_1/sub_5_grad/Reshape
В
1gradients_1/sub_5_grad/tuple/control_dependency_1Identity gradients_1/sub_5_grad/Reshape_1(^gradients_1/sub_5_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_5_grad/Reshape_1*#
_output_shapes
:         
o
gradients_1/sub_3_grad/ShapeShapeStopGradient^Adam*
T0*
out_type0*
_output_shapes
:
t
gradients_1/sub_3_grad/Shape_1Shapemain/q1/Squeeze^Adam*
_output_shapes
:*
T0*
out_type0
└
,gradients_1/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_3_grad/Shapegradients_1/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Й
gradients_1/sub_3_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
gradients_1/sub_3_grad/ReshapeReshapegradients_1/sub_3_grad/Sumgradients_1/sub_3_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
┬
gradients_1/sub_3_grad/Sum_1Sum-gradients_1/pow_grad/tuple/control_dependency.gradients_1/sub_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
b
gradients_1/sub_3_grad/NegNeggradients_1/sub_3_grad/Sum_1*
T0*
_output_shapes
:
Б
 gradients_1/sub_3_grad/Reshape_1Reshapegradients_1/sub_3_grad/Neggradients_1/sub_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
z
'gradients_1/sub_3_grad/tuple/group_depsNoOp^Adam^gradients_1/sub_3_grad/Reshape!^gradients_1/sub_3_grad/Reshape_1
Т
/gradients_1/sub_3_grad/tuple/control_dependencyIdentitygradients_1/sub_3_grad/Reshape(^gradients_1/sub_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_3_grad/Reshape*#
_output_shapes
:         
В
1gradients_1/sub_3_grad/tuple/control_dependency_1Identity gradients_1/sub_3_grad/Reshape_1(^gradients_1/sub_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_3_grad/Reshape_1*#
_output_shapes
:         
o
gradients_1/sub_4_grad/ShapeShapeStopGradient^Adam*
T0*
out_type0*
_output_shapes
:
t
gradients_1/sub_4_grad/Shape_1Shapemain/q2/Squeeze^Adam*
T0*
out_type0*
_output_shapes
:
└
,gradients_1/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_4_grad/Shapegradients_1/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:         :         
└
gradients_1/sub_4_grad/SumSum/gradients_1/pow_1_grad/tuple/control_dependency,gradients_1/sub_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ъ
gradients_1/sub_4_grad/ReshapeReshapegradients_1/sub_4_grad/Sumgradients_1/sub_4_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
─
gradients_1/sub_4_grad/Sum_1Sum/gradients_1/pow_1_grad/tuple/control_dependency.gradients_1/sub_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
b
gradients_1/sub_4_grad/NegNeggradients_1/sub_4_grad/Sum_1*
T0*
_output_shapes
:
Б
 gradients_1/sub_4_grad/Reshape_1Reshapegradients_1/sub_4_grad/Neggradients_1/sub_4_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
z
'gradients_1/sub_4_grad/tuple/group_depsNoOp^Adam^gradients_1/sub_4_grad/Reshape!^gradients_1/sub_4_grad/Reshape_1
Т
/gradients_1/sub_4_grad/tuple/control_dependencyIdentitygradients_1/sub_4_grad/Reshape(^gradients_1/sub_4_grad/tuple/group_deps*#
_output_shapes
:         *
T0*1
_class'
%#loc:@gradients_1/sub_4_grad/Reshape
В
1gradients_1/sub_4_grad/tuple/control_dependency_1Identity gradients_1/sub_4_grad/Reshape_1(^gradients_1/sub_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_4_grad/Reshape_1*#
_output_shapes
:         
ѓ
%gradients_1/main/v/Squeeze_grad/ShapeShapemain/v/dense_2/BiasAdd^Adam*
T0*
out_type0*
_output_shapes
:
╠
'gradients_1/main/v/Squeeze_grad/ReshapeReshape1gradients_1/sub_5_grad/tuple/control_dependency_1%gradients_1/main/v/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ё
&gradients_1/main/q1/Squeeze_grad/ShapeShapemain/q1/dense_2/BiasAdd^Adam*
T0*
out_type0*
_output_shapes
:
╬
(gradients_1/main/q1/Squeeze_grad/ReshapeReshape1gradients_1/sub_3_grad/tuple/control_dependency_1&gradients_1/main/q1/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ё
&gradients_1/main/q2/Squeeze_grad/ShapeShapemain/q2/dense_2/BiasAdd^Adam*
_output_shapes
:*
T0*
out_type0
╬
(gradients_1/main/q2/Squeeze_grad/ReshapeReshape1gradients_1/sub_4_grad/tuple/control_dependency_1&gradients_1/main/q2/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Д
3gradients_1/main/v/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/main/v/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
Д
8gradients_1/main/v/dense_2/BiasAdd_grad/tuple/group_depsNoOp^Adam(^gradients_1/main/v/Squeeze_grad/Reshape4^gradients_1/main/v/dense_2/BiasAdd_grad/BiasAddGrad
ъ
@gradients_1/main/v/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_1/main/v/Squeeze_grad/Reshape9^gradients_1/main/v/dense_2/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_1/main/v/Squeeze_grad/Reshape*'
_output_shapes
:         
Ф
Bgradients_1/main/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_1/main/v/dense_2/BiasAdd_grad/BiasAddGrad9^gradients_1/main/v/dense_2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/main/v/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Е
4gradients_1/main/q1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/main/q1/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
ф
9gradients_1/main/q1/dense_2/BiasAdd_grad/tuple/group_depsNoOp^Adam)^gradients_1/main/q1/Squeeze_grad/Reshape5^gradients_1/main/q1/dense_2/BiasAdd_grad/BiasAddGrad
б
Agradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/main/q1/Squeeze_grad/Reshape:^gradients_1/main/q1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/main/q1/Squeeze_grad/Reshape*'
_output_shapes
:         
»
Cgradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/q1/dense_2/BiasAdd_grad/BiasAddGrad:^gradients_1/main/q1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/main/q1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Е
4gradients_1/main/q2/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/main/q2/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
ф
9gradients_1/main/q2/dense_2/BiasAdd_grad/tuple/group_depsNoOp^Adam)^gradients_1/main/q2/Squeeze_grad/Reshape5^gradients_1/main/q2/dense_2/BiasAdd_grad/BiasAddGrad
б
Agradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/main/q2/Squeeze_grad/Reshape:^gradients_1/main/q2/dense_2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/main/q2/Squeeze_grad/Reshape*'
_output_shapes
:         
»
Cgradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/q2/dense_2/BiasAdd_grad/BiasAddGrad:^gradients_1/main/q2/dense_2/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/main/q2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ь
-gradients_1/main/v/dense_2/MatMul_grad/MatMulMatMul@gradients_1/main/v/dense_2/BiasAdd_grad/tuple/control_dependencymain/v/dense_2/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
Я
/gradients_1/main/v/dense_2/MatMul_grad/MatMul_1MatMulmain/v/dense_1/Relu@gradients_1/main/v/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( 
е
7gradients_1/main/v/dense_2/MatMul_grad/tuple/group_depsNoOp^Adam.^gradients_1/main/v/dense_2/MatMul_grad/MatMul0^gradients_1/main/v/dense_2/MatMul_grad/MatMul_1
Е
?gradients_1/main/v/dense_2/MatMul_grad/tuple/control_dependencyIdentity-gradients_1/main/v/dense_2/MatMul_grad/MatMul8^gradients_1/main/v/dense_2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/main/v/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         ђ
д
Agradients_1/main/v/dense_2/MatMul_grad/tuple/control_dependency_1Identity/gradients_1/main/v/dense_2/MatMul_grad/MatMul_18^gradients_1/main/v/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/main/v/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	ђ
ы
.gradients_1/main/q1/dense_2/MatMul_grad/MatMulMatMulAgradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependencymain/q1/dense_2/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
с
0gradients_1/main/q1/dense_2/MatMul_grad/MatMul_1MatMulmain/q1/dense_1/ReluAgradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( *
T0
Ф
8gradients_1/main/q1/dense_2/MatMul_grad/tuple/group_depsNoOp^Adam/^gradients_1/main/q1/dense_2/MatMul_grad/MatMul1^gradients_1/main/q1/dense_2/MatMul_grad/MatMul_1
Г
@gradients_1/main/q1/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/q1/dense_2/MatMul_grad/MatMul9^gradients_1/main/q1/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         ђ
ф
Bgradients_1/main/q1/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/q1/dense_2/MatMul_grad/MatMul_19^gradients_1/main/q1/dense_2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/main/q1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	ђ
ы
.gradients_1/main/q2/dense_2/MatMul_grad/MatMulMatMulAgradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependencymain/q2/dense_2/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
с
0gradients_1/main/q2/dense_2/MatMul_grad/MatMul_1MatMulmain/q2/dense_1/ReluAgradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( 
Ф
8gradients_1/main/q2/dense_2/MatMul_grad/tuple/group_depsNoOp^Adam/^gradients_1/main/q2/dense_2/MatMul_grad/MatMul1^gradients_1/main/q2/dense_2/MatMul_grad/MatMul_1
Г
@gradients_1/main/q2/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/q2/dense_2/MatMul_grad/MatMul9^gradients_1/main/q2/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q2/dense_2/MatMul_grad/MatMul*(
_output_shapes
:         ђ
ф
Bgradients_1/main/q2/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/q2/dense_2/MatMul_grad/MatMul_19^gradients_1/main/q2/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	ђ*
T0*C
_class9
75loc:@gradients_1/main/q2/dense_2/MatMul_grad/MatMul_1
┬
-gradients_1/main/v/dense_1/Relu_grad/ReluGradReluGrad?gradients_1/main/v/dense_2/MatMul_grad/tuple/control_dependencymain/v/dense_1/Relu*
T0*(
_output_shapes
:         ђ
┼
.gradients_1/main/q1/dense_1/Relu_grad/ReluGradReluGrad@gradients_1/main/q1/dense_2/MatMul_grad/tuple/control_dependencymain/q1/dense_1/Relu*
T0*(
_output_shapes
:         ђ
┼
.gradients_1/main/q2/dense_1/Relu_grad/ReluGradReluGrad@gradients_1/main/q2/dense_2/MatMul_grad/tuple/control_dependencymain/q2/dense_1/Relu*
T0*(
_output_shapes
:         ђ
«
3gradients_1/main/v/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients_1/main/v/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:ђ*
T0
Г
8gradients_1/main/v/dense_1/BiasAdd_grad/tuple/group_depsNoOp^Adam4^gradients_1/main/v/dense_1/BiasAdd_grad/BiasAddGrad.^gradients_1/main/v/dense_1/Relu_grad/ReluGrad
Ф
@gradients_1/main/v/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity-gradients_1/main/v/dense_1/Relu_grad/ReluGrad9^gradients_1/main/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/main/v/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:         ђ
г
Bgradients_1/main/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients_1/main/v/dense_1/BiasAdd_grad/BiasAddGrad9^gradients_1/main/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/main/v/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
░
4gradients_1/main/q1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_1/main/q1/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:ђ
░
9gradients_1/main/q1/dense_1/BiasAdd_grad/tuple/group_depsNoOp^Adam5^gradients_1/main/q1/dense_1/BiasAdd_grad/BiasAddGrad/^gradients_1/main/q1/dense_1/Relu_grad/ReluGrad
»
Agradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients_1/main/q1/dense_1/Relu_grad/ReluGrad:^gradients_1/main/q1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q1/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:         ђ
░
Cgradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/q1/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_1/main/q1/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@gradients_1/main/q1/dense_1/BiasAdd_grad/BiasAddGrad
░
4gradients_1/main/q2/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_1/main/q2/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:ђ
░
9gradients_1/main/q2/dense_1/BiasAdd_grad/tuple/group_depsNoOp^Adam5^gradients_1/main/q2/dense_1/BiasAdd_grad/BiasAddGrad/^gradients_1/main/q2/dense_1/Relu_grad/ReluGrad
»
Agradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients_1/main/q2/dense_1/Relu_grad/ReluGrad:^gradients_1/main/q2/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q2/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:         ђ
░
Cgradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/q2/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_1/main/q2/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@gradients_1/main/q2/dense_1/BiasAdd_grad/BiasAddGrad
Ь
-gradients_1/main/v/dense_1/MatMul_grad/MatMulMatMul@gradients_1/main/v/dense_1/BiasAdd_grad/tuple/control_dependencymain/v/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
▀
/gradients_1/main/v/dense_1/MatMul_grad/MatMul_1MatMulmain/v/dense/Relu@gradients_1/main/v/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
ђђ*
transpose_a(
е
7gradients_1/main/v/dense_1/MatMul_grad/tuple/group_depsNoOp^Adam.^gradients_1/main/v/dense_1/MatMul_grad/MatMul0^gradients_1/main/v/dense_1/MatMul_grad/MatMul_1
Е
?gradients_1/main/v/dense_1/MatMul_grad/tuple/control_dependencyIdentity-gradients_1/main/v/dense_1/MatMul_grad/MatMul8^gradients_1/main/v/dense_1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/main/v/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Д
Agradients_1/main/v/dense_1/MatMul_grad/tuple/control_dependency_1Identity/gradients_1/main/v/dense_1/MatMul_grad/MatMul_18^gradients_1/main/v/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/main/v/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
ы
.gradients_1/main/q1/dense_1/MatMul_grad/MatMulMatMulAgradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependencymain/q1/dense_1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:         ђ*
transpose_a( 
Р
0gradients_1/main/q1/dense_1/MatMul_grad/MatMul_1MatMulmain/q1/dense/ReluAgradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Ф
8gradients_1/main/q1/dense_1/MatMul_grad/tuple/group_depsNoOp^Adam/^gradients_1/main/q1/dense_1/MatMul_grad/MatMul1^gradients_1/main/q1/dense_1/MatMul_grad/MatMul_1
Г
@gradients_1/main/q1/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/q1/dense_1/MatMul_grad/MatMul9^gradients_1/main/q1/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q1/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Ф
Bgradients_1/main/q1/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/q1/dense_1/MatMul_grad/MatMul_19^gradients_1/main/q1/dense_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/main/q1/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
ы
.gradients_1/main/q2/dense_1/MatMul_grad/MatMulMatMulAgradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependencymain/q2/dense_1/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
Р
0gradients_1/main/q2/dense_1/MatMul_grad/MatMul_1MatMulmain/q2/dense/ReluAgradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Ф
8gradients_1/main/q2/dense_1/MatMul_grad/tuple/group_depsNoOp^Adam/^gradients_1/main/q2/dense_1/MatMul_grad/MatMul1^gradients_1/main/q2/dense_1/MatMul_grad/MatMul_1
Г
@gradients_1/main/q2/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/q2/dense_1/MatMul_grad/MatMul9^gradients_1/main/q2/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q2/dense_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Ф
Bgradients_1/main/q2/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/q2/dense_1/MatMul_grad/MatMul_19^gradients_1/main/q2/dense_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/main/q2/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
Й
+gradients_1/main/v/dense/Relu_grad/ReluGradReluGrad?gradients_1/main/v/dense_1/MatMul_grad/tuple/control_dependencymain/v/dense/Relu*(
_output_shapes
:         ђ*
T0
┴
,gradients_1/main/q1/dense/Relu_grad/ReluGradReluGrad@gradients_1/main/q1/dense_1/MatMul_grad/tuple/control_dependencymain/q1/dense/Relu*
T0*(
_output_shapes
:         ђ
┴
,gradients_1/main/q2/dense/Relu_grad/ReluGradReluGrad@gradients_1/main/q2/dense_1/MatMul_grad/tuple/control_dependencymain/q2/dense/Relu*(
_output_shapes
:         ђ*
T0
ф
1gradients_1/main/v/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients_1/main/v/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:ђ*
T0
Д
6gradients_1/main/v/dense/BiasAdd_grad/tuple/group_depsNoOp^Adam2^gradients_1/main/v/dense/BiasAdd_grad/BiasAddGrad,^gradients_1/main/v/dense/Relu_grad/ReluGrad
Б
>gradients_1/main/v/dense/BiasAdd_grad/tuple/control_dependencyIdentity+gradients_1/main/v/dense/Relu_grad/ReluGrad7^gradients_1/main/v/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/main/v/dense/Relu_grad/ReluGrad*(
_output_shapes
:         ђ
ц
@gradients_1/main/v/dense/BiasAdd_grad/tuple/control_dependency_1Identity1gradients_1/main/v/dense/BiasAdd_grad/BiasAddGrad7^gradients_1/main/v/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/main/v/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
г
2gradients_1/main/q1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/main/q1/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:ђ
ф
7gradients_1/main/q1/dense/BiasAdd_grad/tuple/group_depsNoOp^Adam3^gradients_1/main/q1/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/main/q1/dense/Relu_grad/ReluGrad
Д
?gradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/main/q1/dense/Relu_grad/ReluGrad8^gradients_1/main/q1/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*?
_class5
31loc:@gradients_1/main/q1/dense/Relu_grad/ReluGrad
е
Agradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/main/q1/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/main/q1/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*E
_class;
97loc:@gradients_1/main/q1/dense/BiasAdd_grad/BiasAddGrad
г
2gradients_1/main/q2/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/main/q2/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:ђ
ф
7gradients_1/main/q2/dense/BiasAdd_grad/tuple/group_depsNoOp^Adam3^gradients_1/main/q2/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/main/q2/dense/Relu_grad/ReluGrad
Д
?gradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/main/q2/dense/Relu_grad/ReluGrad8^gradients_1/main/q2/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*?
_class5
31loc:@gradients_1/main/q2/dense/Relu_grad/ReluGrad
е
Agradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/main/q2/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/main/q2/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*E
_class;
97loc:@gradients_1/main/q2/dense/BiasAdd_grad/BiasAddGrad
у
+gradients_1/main/v/dense/MatMul_grad/MatMulMatMul>gradients_1/main/v/dense/BiasAdd_grad/tuple/control_dependencymain/v/dense/kernel/read*
T0*'
_output_shapes
:         I*
transpose_a( *
transpose_b(
н
-gradients_1/main/v/dense/MatMul_grad/MatMul_1MatMulPlaceholder>gradients_1/main/v/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	Iђ*
transpose_a(*
transpose_b( 
б
5gradients_1/main/v/dense/MatMul_grad/tuple/group_depsNoOp^Adam,^gradients_1/main/v/dense/MatMul_grad/MatMul.^gradients_1/main/v/dense/MatMul_grad/MatMul_1
а
=gradients_1/main/v/dense/MatMul_grad/tuple/control_dependencyIdentity+gradients_1/main/v/dense/MatMul_grad/MatMul6^gradients_1/main/v/dense/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/main/v/dense/MatMul_grad/MatMul*'
_output_shapes
:         I
ъ
?gradients_1/main/v/dense/MatMul_grad/tuple/control_dependency_1Identity-gradients_1/main/v/dense/MatMul_grad/MatMul_16^gradients_1/main/v/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/main/v/dense/MatMul_grad/MatMul_1*
_output_shapes
:	Iђ
Ж
,gradients_1/main/q1/dense/MatMul_grad/MatMulMatMul?gradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependencymain/q1/dense/kernel/read*'
_output_shapes
:         h*
transpose_a( *
transpose_b(*
T0
┘
.gradients_1/main/q1/dense/MatMul_grad/MatMul_1MatMulmain/q1/concat?gradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	hђ*
transpose_a(
Ц
6gradients_1/main/q1/dense/MatMul_grad/tuple/group_depsNoOp^Adam-^gradients_1/main/q1/dense/MatMul_grad/MatMul/^gradients_1/main/q1/dense/MatMul_grad/MatMul_1
ц
>gradients_1/main/q1/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/main/q1/dense/MatMul_grad/MatMul7^gradients_1/main/q1/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/main/q1/dense/MatMul_grad/MatMul*'
_output_shapes
:         h
б
@gradients_1/main/q1/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/main/q1/dense/MatMul_grad/MatMul_17^gradients_1/main/q1/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	hђ*
T0*A
_class7
53loc:@gradients_1/main/q1/dense/MatMul_grad/MatMul_1
Ж
,gradients_1/main/q2/dense/MatMul_grad/MatMulMatMul?gradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependencymain/q2/dense/kernel/read*'
_output_shapes
:         h*
transpose_a( *
transpose_b(*
T0
┘
.gradients_1/main/q2/dense/MatMul_grad/MatMul_1MatMulmain/q2/concat?gradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	hђ*
transpose_a(
Ц
6gradients_1/main/q2/dense/MatMul_grad/tuple/group_depsNoOp^Adam-^gradients_1/main/q2/dense/MatMul_grad/MatMul/^gradients_1/main/q2/dense/MatMul_grad/MatMul_1
ц
>gradients_1/main/q2/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/main/q2/dense/MatMul_grad/MatMul7^gradients_1/main/q2/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/main/q2/dense/MatMul_grad/MatMul*'
_output_shapes
:         h
б
@gradients_1/main/q2/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/main/q2/dense/MatMul_grad/MatMul_17^gradients_1/main/q2/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q2/dense/MatMul_grad/MatMul_1*
_output_shapes
:	hђ
Є
beta1_power_1/initial_valueConst*
valueB
 *fff?*%
_class
loc:@main/q1/dense/bias*
dtype0*
_output_shapes
: 
ў
beta1_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@main/q1/dense/bias*
	container *
shape: 
╗
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias
u
beta1_power_1/readIdentitybeta1_power_1*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
Є
beta2_power_1/initial_valueConst*
valueB
 *wЙ?*%
_class
loc:@main/q1/dense/bias*
dtype0*
_output_shapes
: 
ў
beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@main/q1/dense/bias*
	container *
shape: 
╗
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: 
u
beta2_power_1/readIdentitybeta2_power_1*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
х
;main/q1/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@main/q1/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:
Ъ
1main/q1/dense/kernel/Adam/Initializer/zeros/ConstConst*'
_class
loc:@main/q1/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ѕ
+main/q1/dense/kernel/Adam/Initializer/zerosFill;main/q1/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1main/q1/dense/kernel/Adam/Initializer/zeros/Const*
T0*'
_class
loc:@main/q1/dense/kernel*

index_type0*
_output_shapes
:	hђ
И
main/q1/dense/kernel/Adam
VariableV2*
shape:	hђ*
dtype0*
_output_shapes
:	hђ*
shared_name *'
_class
loc:@main/q1/dense/kernel*
	container 
Ь
 main/q1/dense/kernel/Adam/AssignAssignmain/q1/dense/kernel/Adam+main/q1/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
ў
main/q1/dense/kernel/Adam/readIdentitymain/q1/dense/kernel/Adam*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	hђ
и
=main/q1/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@main/q1/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:
А
3main/q1/dense/kernel/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@main/q1/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
-main/q1/dense/kernel/Adam_1/Initializer/zerosFill=main/q1/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3main/q1/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	hђ*
T0*'
_class
loc:@main/q1/dense/kernel*

index_type0
║
main/q1/dense/kernel/Adam_1
VariableV2*'
_class
loc:@main/q1/dense/kernel*
	container *
shape:	hђ*
dtype0*
_output_shapes
:	hђ*
shared_name 
З
"main/q1/dense/kernel/Adam_1/AssignAssignmain/q1/dense/kernel/Adam_1-main/q1/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	hђ*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel
ю
 main/q1/dense/kernel/Adam_1/readIdentitymain/q1/dense/kernel/Adam_1*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	hђ
Ф
9main/q1/dense/bias/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
_class
loc:@main/q1/dense/bias*
valueB:ђ
Џ
/main/q1/dense/bias/Adam/Initializer/zeros/ConstConst*%
_class
loc:@main/q1/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
)main/q1/dense/bias/Adam/Initializer/zerosFill9main/q1/dense/bias/Adam/Initializer/zeros/shape_as_tensor/main/q1/dense/bias/Adam/Initializer/zeros/Const*
T0*%
_class
loc:@main/q1/dense/bias*

index_type0*
_output_shapes	
:ђ
г
main/q1/dense/bias/Adam
VariableV2*
shared_name *%
_class
loc:@main/q1/dense/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
Р
main/q1/dense/bias/Adam/AssignAssignmain/q1/dense/bias/Adam)main/q1/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ
ј
main/q1/dense/bias/Adam/readIdentitymain/q1/dense/bias/Adam*
_output_shapes	
:ђ*
T0*%
_class
loc:@main/q1/dense/bias
Г
;main/q1/dense/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
_class
loc:@main/q1/dense/bias*
valueB:ђ
Ю
1main/q1/dense/bias/Adam_1/Initializer/zeros/ConstConst*%
_class
loc:@main/q1/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ѓ
+main/q1/dense/bias/Adam_1/Initializer/zerosFill;main/q1/dense/bias/Adam_1/Initializer/zeros/shape_as_tensor1main/q1/dense/bias/Adam_1/Initializer/zeros/Const*
T0*%
_class
loc:@main/q1/dense/bias*

index_type0*
_output_shapes	
:ђ
«
main/q1/dense/bias/Adam_1
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *%
_class
loc:@main/q1/dense/bias*
	container 
У
 main/q1/dense/bias/Adam_1/AssignAssignmain/q1/dense/bias/Adam_1+main/q1/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ
њ
main/q1/dense/bias/Adam_1/readIdentitymain/q1/dense/bias/Adam_1*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes	
:ђ
╣
=main/q1/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q1/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Б
3main/q1/dense_1/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@main/q1/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
-main/q1/dense_1/kernel/Adam/Initializer/zerosFill=main/q1/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/q1/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@main/q1/dense_1/kernel*

index_type0* 
_output_shapes
:
ђђ
Й
main/q1/dense_1/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@main/q1/dense_1/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ
э
"main/q1/dense_1/kernel/Adam/AssignAssignmain/q1/dense_1/kernel/Adam-main/q1/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel
Ъ
 main/q1/dense_1/kernel/Adam/readIdentitymain/q1/dense_1/kernel/Adam*
T0*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
ђђ
╗
?main/q1/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q1/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
5main/q1/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@main/q1/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
/main/q1/dense_1/kernel/Adam_1/Initializer/zerosFill?main/q1/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/q1/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
ђђ*
T0*)
_class
loc:@main/q1/dense_1/kernel*

index_type0
└
main/q1/dense_1/kernel/Adam_1
VariableV2*)
_class
loc:@main/q1/dense_1/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name 
§
$main/q1/dense_1/kernel/Adam_1/AssignAssignmain/q1/dense_1/kernel/Adam_1/main/q1/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Б
"main/q1/dense_1/kernel/Adam_1/readIdentitymain/q1/dense_1/kernel/Adam_1*
T0*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
ђђ
Б
+main/q1/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q1/dense_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
░
main/q1/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *'
_class
loc:@main/q1/dense_1/bias*
	container *
shape:ђ
Ж
 main/q1/dense_1/bias/Adam/AssignAssignmain/q1/dense_1/bias/Adam+main/q1/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias
ћ
main/q1/dense_1/bias/Adam/readIdentitymain/q1/dense_1/bias/Adam*
T0*'
_class
loc:@main/q1/dense_1/bias*
_output_shapes	
:ђ
Ц
-main/q1/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q1/dense_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
▓
main/q1/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *'
_class
loc:@main/q1/dense_1/bias*
	container *
shape:ђ
­
"main/q1/dense_1/bias/Adam_1/AssignAssignmain/q1/dense_1/bias/Adam_1-main/q1/dense_1/bias/Adam_1/Initializer/zeros*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
ў
 main/q1/dense_1/bias/Adam_1/readIdentitymain/q1/dense_1/bias/Adam_1*
T0*'
_class
loc:@main/q1/dense_1/bias*
_output_shapes	
:ђ
»
-main/q1/dense_2/kernel/Adam/Initializer/zerosConst*)
_class
loc:@main/q1/dense_2/kernel*
valueB	ђ*    *
dtype0*
_output_shapes
:	ђ
╝
main/q1/dense_2/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	ђ*
shared_name *)
_class
loc:@main/q1/dense_2/kernel*
	container *
shape:	ђ
Ш
"main/q1/dense_2/kernel/Adam/AssignAssignmain/q1/dense_2/kernel/Adam-main/q1/dense_2/kernel/Adam/Initializer/zeros*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
ъ
 main/q1/dense_2/kernel/Adam/readIdentitymain/q1/dense_2/kernel/Adam*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/q1/dense_2/kernel
▒
/main/q1/dense_2/kernel/Adam_1/Initializer/zerosConst*)
_class
loc:@main/q1/dense_2/kernel*
valueB	ђ*    *
dtype0*
_output_shapes
:	ђ
Й
main/q1/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	ђ*
shared_name *)
_class
loc:@main/q1/dense_2/kernel*
	container *
shape:	ђ
Ч
$main/q1/dense_2/kernel/Adam_1/AssignAssignmain/q1/dense_2/kernel/Adam_1/main/q1/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel
б
"main/q1/dense_2/kernel/Adam_1/readIdentitymain/q1/dense_2/kernel/Adam_1*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
:	ђ
А
+main/q1/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q1/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
«
main/q1/dense_2/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/q1/dense_2/bias*
	container 
ж
 main/q1/dense_2/bias/Adam/AssignAssignmain/q1/dense_2/bias/Adam+main/q1/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
Њ
main/q1/dense_2/bias/Adam/readIdentitymain/q1/dense_2/bias/Adam*
_output_shapes
:*
T0*'
_class
loc:@main/q1/dense_2/bias
Б
-main/q1/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q1/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
░
main/q1/dense_2/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@main/q1/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
№
"main/q1/dense_2/bias/Adam_1/AssignAssignmain/q1/dense_2/bias/Adam_1-main/q1/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
Ќ
 main/q1/dense_2/bias/Adam_1/readIdentitymain/q1/dense_2/bias/Adam_1*
T0*'
_class
loc:@main/q1/dense_2/bias*
_output_shapes
:
х
;main/q2/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@main/q2/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:
Ъ
1main/q2/dense/kernel/Adam/Initializer/zeros/ConstConst*'
_class
loc:@main/q2/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ѕ
+main/q2/dense/kernel/Adam/Initializer/zerosFill;main/q2/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1main/q2/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	hђ*
T0*'
_class
loc:@main/q2/dense/kernel*

index_type0
И
main/q2/dense/kernel/Adam
VariableV2*'
_class
loc:@main/q2/dense/kernel*
	container *
shape:	hђ*
dtype0*
_output_shapes
:	hђ*
shared_name 
Ь
 main/q2/dense/kernel/Adam/AssignAssignmain/q2/dense/kernel/Adam+main/q2/dense/kernel/Adam/Initializer/zeros*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes
:	hђ*
use_locking(
ў
main/q2/dense/kernel/Adam/readIdentitymain/q2/dense/kernel/Adam*
_output_shapes
:	hђ*
T0*'
_class
loc:@main/q2/dense/kernel
и
=main/q2/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@main/q2/dense/kernel*
valueB"h      *
dtype0*
_output_shapes
:
А
3main/q2/dense/kernel/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@main/q2/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
-main/q2/dense/kernel/Adam_1/Initializer/zerosFill=main/q2/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3main/q2/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*'
_class
loc:@main/q2/dense/kernel*

index_type0*
_output_shapes
:	hђ
║
main/q2/dense/kernel/Adam_1
VariableV2*
shape:	hђ*
dtype0*
_output_shapes
:	hђ*
shared_name *'
_class
loc:@main/q2/dense/kernel*
	container 
З
"main/q2/dense/kernel/Adam_1/AssignAssignmain/q2/dense/kernel/Adam_1-main/q2/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
ю
 main/q2/dense/kernel/Adam_1/readIdentitymain/q2/dense/kernel/Adam_1*
_output_shapes
:	hђ*
T0*'
_class
loc:@main/q2/dense/kernel
Ф
9main/q2/dense/bias/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
_class
loc:@main/q2/dense/bias*
valueB:ђ
Џ
/main/q2/dense/bias/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *%
_class
loc:@main/q2/dense/bias*
valueB
 *    
Ч
)main/q2/dense/bias/Adam/Initializer/zerosFill9main/q2/dense/bias/Adam/Initializer/zeros/shape_as_tensor/main/q2/dense/bias/Adam/Initializer/zeros/Const*
T0*%
_class
loc:@main/q2/dense/bias*

index_type0*
_output_shapes	
:ђ
г
main/q2/dense/bias/Adam
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *%
_class
loc:@main/q2/dense/bias*
	container 
Р
main/q2/dense/bias/Adam/AssignAssignmain/q2/dense/bias/Adam)main/q2/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes	
:ђ
ј
main/q2/dense/bias/Adam/readIdentitymain/q2/dense/bias/Adam*
T0*%
_class
loc:@main/q2/dense/bias*
_output_shapes	
:ђ
Г
;main/q2/dense/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*%
_class
loc:@main/q2/dense/bias*
valueB:ђ*
dtype0*
_output_shapes
:
Ю
1main/q2/dense/bias/Adam_1/Initializer/zeros/ConstConst*%
_class
loc:@main/q2/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ѓ
+main/q2/dense/bias/Adam_1/Initializer/zerosFill;main/q2/dense/bias/Adam_1/Initializer/zeros/shape_as_tensor1main/q2/dense/bias/Adam_1/Initializer/zeros/Const*
T0*%
_class
loc:@main/q2/dense/bias*

index_type0*
_output_shapes	
:ђ
«
main/q2/dense/bias/Adam_1
VariableV2*%
_class
loc:@main/q2/dense/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
У
 main/q2/dense/bias/Adam_1/AssignAssignmain/q2/dense/bias/Adam_1+main/q2/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes	
:ђ
њ
main/q2/dense/bias/Adam_1/readIdentitymain/q2/dense/bias/Adam_1*
T0*%
_class
loc:@main/q2/dense/bias*
_output_shapes	
:ђ
╣
=main/q2/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*)
_class
loc:@main/q2/dense_1/kernel*
valueB"      
Б
3main/q2/dense_1/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@main/q2/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
-main/q2/dense_1/kernel/Adam/Initializer/zerosFill=main/q2/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/q2/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@main/q2/dense_1/kernel*

index_type0* 
_output_shapes
:
ђђ
Й
main/q2/dense_1/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *)
_class
loc:@main/q2/dense_1/kernel*
	container *
shape:
ђђ
э
"main/q2/dense_1/kernel/Adam/AssignAssignmain/q2/dense_1/kernel/Adam-main/q2/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Ъ
 main/q2/dense_1/kernel/Adam/readIdentitymain/q2/dense_1/kernel/Adam*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:
ђђ
╗
?main/q2/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q2/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ц
5main/q2/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@main/q2/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
/main/q2/dense_1/kernel/Adam_1/Initializer/zerosFill?main/q2/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/q2/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@main/q2/dense_1/kernel*

index_type0* 
_output_shapes
:
ђђ
└
main/q2/dense_1/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *)
_class
loc:@main/q2/dense_1/kernel*
	container *
shape:
ђђ
§
$main/q2/dense_1/kernel/Adam_1/AssignAssignmain/q2/dense_1/kernel/Adam_1/main/q2/dense_1/kernel/Adam_1/Initializer/zeros*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
Б
"main/q2/dense_1/kernel/Adam_1/readIdentitymain/q2/dense_1/kernel/Adam_1*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:
ђђ
Б
+main/q2/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q2/dense_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
░
main/q2/dense_1/bias/Adam
VariableV2*'
_class
loc:@main/q2/dense_1/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
Ж
 main/q2/dense_1/bias/Adam/AssignAssignmain/q2/dense_1/bias/Adam+main/q2/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias
ћ
main/q2/dense_1/bias/Adam/readIdentitymain/q2/dense_1/bias/Adam*
T0*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:ђ
Ц
-main/q2/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q2/dense_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
▓
main/q2/dense_1/bias/Adam_1
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *'
_class
loc:@main/q2/dense_1/bias
­
"main/q2/dense_1/bias/Adam_1/AssignAssignmain/q2/dense_1/bias/Adam_1-main/q2/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
ў
 main/q2/dense_1/bias/Adam_1/readIdentitymain/q2/dense_1/bias/Adam_1*
T0*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:ђ
»
-main/q2/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	ђ*)
_class
loc:@main/q2/dense_2/kernel*
valueB	ђ*    
╝
main/q2/dense_2/kernel/Adam
VariableV2*)
_class
loc:@main/q2/dense_2/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name 
Ш
"main/q2/dense_2/kernel/Adam/AssignAssignmain/q2/dense_2/kernel/Adam-main/q2/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
ъ
 main/q2/dense_2/kernel/Adam/readIdentitymain/q2/dense_2/kernel/Adam*
_output_shapes
:	ђ*
T0*)
_class
loc:@main/q2/dense_2/kernel
▒
/main/q2/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	ђ*)
_class
loc:@main/q2/dense_2/kernel*
valueB	ђ*    
Й
main/q2/dense_2/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@main/q2/dense_2/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
Ч
$main/q2/dense_2/kernel/Adam_1/AssignAssignmain/q2/dense_2/kernel/Adam_1/main/q2/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel
б
"main/q2/dense_2/kernel/Adam_1/readIdentitymain/q2/dense_2/kernel/Adam_1*
T0*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
:	ђ
А
+main/q2/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q2/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
«
main/q2/dense_2/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/q2/dense_2/bias*
	container 
ж
 main/q2/dense_2/bias/Adam/AssignAssignmain/q2/dense_2/bias/Adam+main/q2/dense_2/bias/Adam/Initializer/zeros*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Њ
main/q2/dense_2/bias/Adam/readIdentitymain/q2/dense_2/bias/Adam*
T0*'
_class
loc:@main/q2/dense_2/bias*
_output_shapes
:
Б
-main/q2/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q2/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
░
main/q2/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/q2/dense_2/bias*
	container *
shape:
№
"main/q2/dense_2/bias/Adam_1/AssignAssignmain/q2/dense_2/bias/Adam_1-main/q2/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
Ќ
 main/q2/dense_2/bias/Adam_1/readIdentitymain/q2/dense_2/bias/Adam_1*
_output_shapes
:*
T0*'
_class
loc:@main/q2/dense_2/bias
│
:main/v/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/v/dense/kernel*
valueB"I      *
dtype0*
_output_shapes
:
Ю
0main/v/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/v/dense/kernel*
valueB
 *    
ё
*main/v/dense/kernel/Adam/Initializer/zerosFill:main/v/dense/kernel/Adam/Initializer/zeros/shape_as_tensor0main/v/dense/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/v/dense/kernel*

index_type0*
_output_shapes
:	Iђ
Х
main/v/dense/kernel/Adam
VariableV2*&
_class
loc:@main/v/dense/kernel*
	container *
shape:	Iђ*
dtype0*
_output_shapes
:	Iђ*
shared_name 
Ж
main/v/dense/kernel/Adam/AssignAssignmain/v/dense/kernel/Adam*main/v/dense/kernel/Adam/Initializer/zeros*
T0*&
_class
loc:@main/v/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ*
use_locking(
Ћ
main/v/dense/kernel/Adam/readIdentitymain/v/dense/kernel/Adam*
T0*&
_class
loc:@main/v/dense/kernel*
_output_shapes
:	Iђ
х
<main/v/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/v/dense/kernel*
valueB"I      *
dtype0*
_output_shapes
:
Ъ
2main/v/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/v/dense/kernel*
valueB
 *    
і
,main/v/dense/kernel/Adam_1/Initializer/zerosFill<main/v/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/v/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/v/dense/kernel*

index_type0*
_output_shapes
:	Iђ
И
main/v/dense/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/v/dense/kernel*
	container *
shape:	Iђ*
dtype0*
_output_shapes
:	Iђ
­
!main/v/dense/kernel/Adam_1/AssignAssignmain/v/dense/kernel/Adam_1,main/v/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/v/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
Ў
main/v/dense/kernel/Adam_1/readIdentitymain/v/dense/kernel/Adam_1*
_output_shapes
:	Iђ*
T0*&
_class
loc:@main/v/dense/kernel
Е
8main/v/dense/bias/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@main/v/dense/bias*
valueB:ђ*
dtype0*
_output_shapes
:
Ў
.main/v/dense/bias/Adam/Initializer/zeros/ConstConst*$
_class
loc:@main/v/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
(main/v/dense/bias/Adam/Initializer/zerosFill8main/v/dense/bias/Adam/Initializer/zeros/shape_as_tensor.main/v/dense/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:ђ*
T0*$
_class
loc:@main/v/dense/bias*

index_type0
ф
main/v/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *$
_class
loc:@main/v/dense/bias*
	container *
shape:ђ
я
main/v/dense/bias/Adam/AssignAssignmain/v/dense/bias/Adam(main/v/dense/bias/Adam/Initializer/zeros*
T0*$
_class
loc:@main/v/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
І
main/v/dense/bias/Adam/readIdentitymain/v/dense/bias/Adam*
_output_shapes	
:ђ*
T0*$
_class
loc:@main/v/dense/bias
Ф
:main/v/dense/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@main/v/dense/bias*
valueB:ђ*
dtype0*
_output_shapes
:
Џ
0main/v/dense/bias/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@main/v/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
■
*main/v/dense/bias/Adam_1/Initializer/zerosFill:main/v/dense/bias/Adam_1/Initializer/zeros/shape_as_tensor0main/v/dense/bias/Adam_1/Initializer/zeros/Const*
T0*$
_class
loc:@main/v/dense/bias*

index_type0*
_output_shapes	
:ђ
г
main/v/dense/bias/Adam_1
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *$
_class
loc:@main/v/dense/bias
С
main/v/dense/bias/Adam_1/AssignAssignmain/v/dense/bias/Adam_1*main/v/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/v/dense/bias*
validate_shape(*
_output_shapes	
:ђ
Ј
main/v/dense/bias/Adam_1/readIdentitymain/v/dense/bias/Adam_1*
T0*$
_class
loc:@main/v/dense/bias*
_output_shapes	
:ђ
и
<main/v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@main/v/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
А
2main/v/dense_1/kernel/Adam/Initializer/zeros/ConstConst*(
_class
loc:@main/v/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
,main/v/dense_1/kernel/Adam/Initializer/zerosFill<main/v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor2main/v/dense_1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
ђђ*
T0*(
_class
loc:@main/v/dense_1/kernel*

index_type0
╝
main/v/dense_1/kernel/Adam
VariableV2*
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *(
_class
loc:@main/v/dense_1/kernel*
	container 
з
!main/v/dense_1/kernel/Adam/AssignAssignmain/v/dense_1/kernel/Adam,main/v/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*(
_class
loc:@main/v/dense_1/kernel
ю
main/v/dense_1/kernel/Adam/readIdentitymain/v/dense_1/kernel/Adam*
T0*(
_class
loc:@main/v/dense_1/kernel* 
_output_shapes
:
ђђ
╣
>main/v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@main/v/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Б
4main/v/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *(
_class
loc:@main/v/dense_1/kernel*
valueB
 *    
Њ
.main/v/dense_1/kernel/Adam_1/Initializer/zerosFill>main/v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor4main/v/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*(
_class
loc:@main/v/dense_1/kernel*

index_type0* 
_output_shapes
:
ђђ
Й
main/v/dense_1/kernel/Adam_1
VariableV2*(
_class
loc:@main/v/dense_1/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name 
щ
#main/v/dense_1/kernel/Adam_1/AssignAssignmain/v/dense_1/kernel/Adam_1.main/v/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@main/v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
а
!main/v/dense_1/kernel/Adam_1/readIdentitymain/v/dense_1/kernel/Adam_1*
T0*(
_class
loc:@main/v/dense_1/kernel* 
_output_shapes
:
ђђ
А
*main/v/dense_1/bias/Adam/Initializer/zerosConst*&
_class
loc:@main/v/dense_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
«
main/v/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *&
_class
loc:@main/v/dense_1/bias*
	container *
shape:ђ
Т
main/v/dense_1/bias/Adam/AssignAssignmain/v/dense_1/bias/Adam*main/v/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
Љ
main/v/dense_1/bias/Adam/readIdentitymain/v/dense_1/bias/Adam*
T0*&
_class
loc:@main/v/dense_1/bias*
_output_shapes	
:ђ
Б
,main/v/dense_1/bias/Adam_1/Initializer/zerosConst*&
_class
loc:@main/v/dense_1/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
░
main/v/dense_1/bias/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/v/dense_1/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
В
!main/v/dense_1/bias/Adam_1/AssignAssignmain/v/dense_1/bias/Adam_1,main/v/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
Ћ
main/v/dense_1/bias/Adam_1/readIdentitymain/v/dense_1/bias/Adam_1*
T0*&
_class
loc:@main/v/dense_1/bias*
_output_shapes	
:ђ
Г
,main/v/dense_2/kernel/Adam/Initializer/zerosConst*(
_class
loc:@main/v/dense_2/kernel*
valueB	ђ*    *
dtype0*
_output_shapes
:	ђ
║
main/v/dense_2/kernel/Adam
VariableV2*
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name *(
_class
loc:@main/v/dense_2/kernel*
	container 
Ы
!main/v/dense_2/kernel/Adam/AssignAssignmain/v/dense_2/kernel/Adam,main/v/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*(
_class
loc:@main/v/dense_2/kernel
Џ
main/v/dense_2/kernel/Adam/readIdentitymain/v/dense_2/kernel/Adam*
T0*(
_class
loc:@main/v/dense_2/kernel*
_output_shapes
:	ђ
»
.main/v/dense_2/kernel/Adam_1/Initializer/zerosConst*(
_class
loc:@main/v/dense_2/kernel*
valueB	ђ*    *
dtype0*
_output_shapes
:	ђ
╝
main/v/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	ђ*
shared_name *(
_class
loc:@main/v/dense_2/kernel*
	container *
shape:	ђ
Э
#main/v/dense_2/kernel/Adam_1/AssignAssignmain/v/dense_2/kernel/Adam_1.main/v/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*(
_class
loc:@main/v/dense_2/kernel
Ъ
!main/v/dense_2/kernel/Adam_1/readIdentitymain/v/dense_2/kernel/Adam_1*
T0*(
_class
loc:@main/v/dense_2/kernel*
_output_shapes
:	ђ
Ъ
*main/v/dense_2/bias/Adam/Initializer/zerosConst*&
_class
loc:@main/v/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
г
main/v/dense_2/bias/Adam
VariableV2*
shared_name *&
_class
loc:@main/v/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
т
main/v/dense_2/bias/Adam/AssignAssignmain/v/dense_2/bias/Adam*main/v/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@main/v/dense_2/bias
љ
main/v/dense_2/bias/Adam/readIdentitymain/v/dense_2/bias/Adam*
_output_shapes
:*
T0*&
_class
loc:@main/v/dense_2/bias
А
,main/v/dense_2/bias/Adam_1/Initializer/zerosConst*&
_class
loc:@main/v/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
«
main/v/dense_2/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *&
_class
loc:@main/v/dense_2/bias
в
!main/v/dense_2/bias/Adam_1/AssignAssignmain/v/dense_2/bias/Adam_1,main/v/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@main/v/dense_2/bias
ћ
main/v/dense_2/bias/Adam_1/readIdentitymain/v/dense_2/bias/Adam_1*
T0*&
_class
loc:@main/v/dense_2/bias*
_output_shapes
:
`
Adam_1/learning_rateConst^Adam*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
X
Adam_1/beta1Const^Adam*
valueB
 *fff?*
dtype0*
_output_shapes
: 
X
Adam_1/beta2Const^Adam*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
Z
Adam_1/epsilonConst^Adam*
dtype0*
_output_shapes
: *
valueB
 *w╠+2
Г
,Adam_1/update_main/q1/dense/kernel/ApplyAdam	ApplyAdammain/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/main/q1/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/q1/dense/kernel*
use_nesterov( *
_output_shapes
:	hђ
а
*Adam_1/update_main/q1/dense/bias/ApplyAdam	ApplyAdammain/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@main/q1/dense/bias*
use_nesterov( *
_output_shapes	
:ђ
║
.Adam_1/update_main/q1/dense_1/kernel/ApplyAdam	ApplyAdammain/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
ђђ*
use_locking( *
T0*)
_class
loc:@main/q1/dense_1/kernel
г
,Adam_1/update_main/q1/dense_1/bias/ApplyAdam	ApplyAdammain/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@main/q1/dense_1/bias*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( 
╣
.Adam_1/update_main/q1/dense_2/kernel/ApplyAdam	ApplyAdammain/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q1/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@main/q1/dense_2/kernel*
use_nesterov( *
_output_shapes
:	ђ
Ф
,Adam_1/update_main/q1/dense_2/bias/ApplyAdam	ApplyAdammain/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/q1/dense_2/bias*
use_nesterov( *
_output_shapes
:
Г
,Adam_1/update_main/q2/dense/kernel/ApplyAdam	ApplyAdammain/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/main/q2/dense/MatMul_grad/tuple/control_dependency_1*
T0*'
_class
loc:@main/q2/dense/kernel*
use_nesterov( *
_output_shapes
:	hђ*
use_locking( 
а
*Adam_1/update_main/q2/dense/bias/ApplyAdam	ApplyAdammain/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@main/q2/dense/bias*
use_nesterov( *
_output_shapes	
:ђ
║
.Adam_1/update_main/q2/dense_1/kernel/ApplyAdam	ApplyAdammain/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q2/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
ђђ*
use_locking( *
T0*)
_class
loc:@main/q2/dense_1/kernel
г
,Adam_1/update_main/q2/dense_1/bias/ApplyAdam	ApplyAdammain/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0*'
_class
loc:@main/q2/dense_1/bias
╣
.Adam_1/update_main/q2/dense_2/kernel/ApplyAdam	ApplyAdammain/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q2/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@main/q2/dense_2/kernel*
use_nesterov( *
_output_shapes
:	ђ
Ф
,Adam_1/update_main/q2/dense_2/bias/ApplyAdam	ApplyAdammain/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/q2/dense_2/bias*
use_nesterov( *
_output_shapes
:
Д
+Adam_1/update_main/v/dense/kernel/ApplyAdam	ApplyAdammain/v/dense/kernelmain/v/dense/kernel/Adammain/v/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon?gradients_1/main/v/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/v/dense/kernel*
use_nesterov( *
_output_shapes
:	Iђ
џ
)Adam_1/update_main/v/dense/bias/ApplyAdam	ApplyAdammain/v/dense/biasmain/v/dense/bias/Adammain/v/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/main/v/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0*$
_class
loc:@main/v/dense/bias
┤
-Adam_1/update_main/v/dense_1/kernel/ApplyAdam	ApplyAdammain/v/dense_1/kernelmain/v/dense_1/kernel/Adammain/v/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/v/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
ђђ*
use_locking( *
T0*(
_class
loc:@main/v/dense_1/kernel
д
+Adam_1/update_main/v/dense_1/bias/ApplyAdam	ApplyAdammain/v/dense_1/biasmain/v/dense_1/bias/Adammain/v/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/v/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/v/dense_1/bias*
use_nesterov( *
_output_shapes	
:ђ
│
-Adam_1/update_main/v/dense_2/kernel/ApplyAdam	ApplyAdammain/v/dense_2/kernelmain/v/dense_2/kernel/Adammain/v/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/v/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@main/v/dense_2/kernel*
use_nesterov( *
_output_shapes
:	ђ
Ц
+Adam_1/update_main/v/dense_2/bias/ApplyAdam	ApplyAdammain/v/dense_2/biasmain/v/dense_2/bias/Adammain/v/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/v/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/v/dense_2/bias*
use_nesterov( *
_output_shapes
:
╔

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1+^Adam_1/update_main/q1/dense/bias/ApplyAdam-^Adam_1/update_main/q1/dense/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_1/bias/ApplyAdam/^Adam_1/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_2/bias/ApplyAdam/^Adam_1/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_1/update_main/q2/dense/bias/ApplyAdam-^Adam_1/update_main/q2/dense/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_1/bias/ApplyAdam/^Adam_1/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_2/bias/ApplyAdam/^Adam_1/update_main/q2/dense_2/kernel/ApplyAdam*^Adam_1/update_main/v/dense/bias/ApplyAdam,^Adam_1/update_main/v/dense/kernel/ApplyAdam,^Adam_1/update_main/v/dense_1/bias/ApplyAdam.^Adam_1/update_main/v/dense_1/kernel/ApplyAdam,^Adam_1/update_main/v/dense_2/bias/ApplyAdam.^Adam_1/update_main/v/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0*%
_class
loc:@main/q1/dense/bias
Б
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: 
╦
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2+^Adam_1/update_main/q1/dense/bias/ApplyAdam-^Adam_1/update_main/q1/dense/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_1/bias/ApplyAdam/^Adam_1/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_2/bias/ApplyAdam/^Adam_1/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_1/update_main/q2/dense/bias/ApplyAdam-^Adam_1/update_main/q2/dense/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_1/bias/ApplyAdam/^Adam_1/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_2/bias/ApplyAdam/^Adam_1/update_main/q2/dense_2/kernel/ApplyAdam*^Adam_1/update_main/v/dense/bias/ApplyAdam,^Adam_1/update_main/v/dense/kernel/ApplyAdam,^Adam_1/update_main/v/dense_1/bias/ApplyAdam.^Adam_1/update_main/v/dense_1/kernel/ApplyAdam,^Adam_1/update_main/v/dense_2/bias/ApplyAdam.^Adam_1/update_main/v/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
Д
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: 
Ё
Adam_1NoOp^Adam^Adam_1/Assign^Adam_1/Assign_1+^Adam_1/update_main/q1/dense/bias/ApplyAdam-^Adam_1/update_main/q1/dense/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_1/bias/ApplyAdam/^Adam_1/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_2/bias/ApplyAdam/^Adam_1/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_1/update_main/q2/dense/bias/ApplyAdam-^Adam_1/update_main/q2/dense/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_1/bias/ApplyAdam/^Adam_1/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_2/bias/ApplyAdam/^Adam_1/update_main/q2/dense_2/kernel/ApplyAdam*^Adam_1/update_main/v/dense/bias/ApplyAdam,^Adam_1/update_main/v/dense/kernel/ApplyAdam,^Adam_1/update_main/v/dense_1/bias/ApplyAdam.^Adam_1/update_main/v/dense_1/kernel/ApplyAdam,^Adam_1/update_main/v/dense_2/bias/ApplyAdam.^Adam_1/update_main/v/dense_2/kernel/ApplyAdam
U
mul_7/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
\
mul_7Mulmul_7/xtarget/pi/dense/kernel/read*
T0*
_output_shapes
:	Iђ
U
mul_8/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
Z
mul_8Mulmul_8/xmain/pi/dense/kernel/read*
T0*
_output_shapes
:	Iђ
D
add_3Addmul_7mul_8*
_output_shapes
:	Iђ*
T0
Г
AssignAssigntarget/pi/dense/kerneladd_3*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
U
mul_9/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
V
mul_9Mulmul_9/xtarget/pi/dense/bias/read*
T0*
_output_shapes	
:ђ
V
mul_10/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
V
mul_10Mulmul_10/xmain/pi/dense/bias/read*
T0*
_output_shapes	
:ђ
A
add_4Addmul_9mul_10*
T0*
_output_shapes	
:ђ
Д
Assign_1Assigntarget/pi/dense/biasadd_4*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
V
mul_11/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
a
mul_11Mulmul_11/xtarget/pi/dense_1/kernel/read*
T0* 
_output_shapes
:
ђђ
V
mul_12/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *
ОБ;
_
mul_12Mulmul_12/xmain/pi/dense_1/kernel/read* 
_output_shapes
:
ђђ*
T0
G
add_5Addmul_11mul_12* 
_output_shapes
:
ђђ*
T0
┤
Assign_2Assigntarget/pi/dense_1/kerneladd_5*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
V
mul_13/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *RИ~?
Z
mul_13Mulmul_13/xtarget/pi/dense_1/bias/read*
_output_shapes	
:ђ*
T0
V
mul_14/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
X
mul_14Mulmul_14/xmain/pi/dense_1/bias/read*
_output_shapes	
:ђ*
T0
B
add_6Addmul_13mul_14*
T0*
_output_shapes	
:ђ
Ф
Assign_3Assigntarget/pi/dense_1/biasadd_6*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
V
mul_15/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
`
mul_15Mulmul_15/xtarget/pi/dense_2/kernel/read*
T0*
_output_shapes
:	ђ
V
mul_16/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
^
mul_16Mulmul_16/xmain/pi/dense_2/kernel/read*
T0*
_output_shapes
:	ђ
F
add_7Addmul_15mul_16*
T0*
_output_shapes
:	ђ
│
Assign_4Assigntarget/pi/dense_2/kerneladd_7*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
V
mul_17/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
Y
mul_17Mulmul_17/xtarget/pi/dense_2/bias/read*
_output_shapes
:*
T0
V
mul_18/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
W
mul_18Mulmul_18/xmain/pi/dense_2/bias/read*
T0*
_output_shapes
:
A
add_8Addmul_17mul_18*
T0*
_output_shapes
:
ф
Assign_5Assigntarget/pi/dense_2/biasadd_8*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
V
mul_19/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
`
mul_19Mulmul_19/xtarget/pi/dense_3/kernel/read*
T0*
_output_shapes
:	ђ
V
mul_20/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
^
mul_20Mulmul_20/xmain/pi/dense_3/kernel/read*
T0*
_output_shapes
:	ђ
F
add_9Addmul_19mul_20*
T0*
_output_shapes
:	ђ
│
Assign_6Assigntarget/pi/dense_3/kerneladd_9*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ
V
mul_21/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
Y
mul_21Mulmul_21/xtarget/pi/dense_3/bias/read*
T0*
_output_shapes
:
V
mul_22/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
W
mul_22Mulmul_22/xmain/pi/dense_3/bias/read*
T0*
_output_shapes
:
B
add_10Addmul_21mul_22*
T0*
_output_shapes
:
Ф
Assign_7Assigntarget/pi/dense_3/biasadd_10*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/pi/dense_3/bias
V
mul_23/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
^
mul_23Mulmul_23/xtarget/q1/dense/kernel/read*
T0*
_output_shapes
:	hђ
V
mul_24/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
\
mul_24Mulmul_24/xmain/q1/dense/kernel/read*
T0*
_output_shapes
:	hђ
G
add_11Addmul_23mul_24*
T0*
_output_shapes
:	hђ
░
Assign_8Assigntarget/q1/dense/kerneladd_11*
use_locking(*
T0*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
V
mul_25/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
X
mul_25Mulmul_25/xtarget/q1/dense/bias/read*
T0*
_output_shapes	
:ђ
V
mul_26/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
V
mul_26Mulmul_26/xmain/q1/dense/bias/read*
T0*
_output_shapes	
:ђ
C
add_12Addmul_25mul_26*
T0*
_output_shapes	
:ђ
е
Assign_9Assigntarget/q1/dense/biasadd_12*
T0*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
V
mul_27/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
a
mul_27Mulmul_27/xtarget/q1/dense_1/kernel/read*
T0* 
_output_shapes
:
ђђ
V
mul_28/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *
ОБ;
_
mul_28Mulmul_28/xmain/q1/dense_1/kernel/read*
T0* 
_output_shapes
:
ђђ
H
add_13Addmul_27mul_28*
T0* 
_output_shapes
:
ђђ
Х
	Assign_10Assigntarget/q1/dense_1/kerneladd_13*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
V
mul_29/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
Z
mul_29Mulmul_29/xtarget/q1/dense_1/bias/read*
_output_shapes	
:ђ*
T0
V
mul_30/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
X
mul_30Mulmul_30/xmain/q1/dense_1/bias/read*
T0*
_output_shapes	
:ђ
C
add_14Addmul_29mul_30*
_output_shapes	
:ђ*
T0
Г
	Assign_11Assigntarget/q1/dense_1/biasadd_14*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
V
mul_31/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
`
mul_31Mulmul_31/xtarget/q1/dense_2/kernel/read*
T0*
_output_shapes
:	ђ
V
mul_32/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
^
mul_32Mulmul_32/xmain/q1/dense_2/kernel/read*
_output_shapes
:	ђ*
T0
G
add_15Addmul_31mul_32*
T0*
_output_shapes
:	ђ
х
	Assign_12Assigntarget/q1/dense_2/kerneladd_15*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_2/kernel
V
mul_33/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
Y
mul_33Mulmul_33/xtarget/q1/dense_2/bias/read*
T0*
_output_shapes
:
V
mul_34/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
W
mul_34Mulmul_34/xmain/q1/dense_2/bias/read*
T0*
_output_shapes
:
B
add_16Addmul_33mul_34*
_output_shapes
:*
T0
г
	Assign_13Assigntarget/q1/dense_2/biasadd_16*
T0*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
V
mul_35/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
^
mul_35Mulmul_35/xtarget/q2/dense/kernel/read*
_output_shapes
:	hђ*
T0
V
mul_36/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
\
mul_36Mulmul_36/xmain/q2/dense/kernel/read*
T0*
_output_shapes
:	hђ
G
add_17Addmul_35mul_36*
T0*
_output_shapes
:	hђ
▒
	Assign_14Assigntarget/q2/dense/kerneladd_17*
validate_shape(*
_output_shapes
:	hђ*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel
V
mul_37/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *RИ~?
X
mul_37Mulmul_37/xtarget/q2/dense/bias/read*
T0*
_output_shapes	
:ђ
V
mul_38/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
V
mul_38Mulmul_38/xmain/q2/dense/bias/read*
T0*
_output_shapes	
:ђ
C
add_18Addmul_37mul_38*
T0*
_output_shapes	
:ђ
Е
	Assign_15Assigntarget/q2/dense/biasadd_18*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*'
_class
loc:@target/q2/dense/bias
V
mul_39/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
a
mul_39Mulmul_39/xtarget/q2/dense_1/kernel/read* 
_output_shapes
:
ђђ*
T0
V
mul_40/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *
ОБ;
_
mul_40Mulmul_40/xmain/q2/dense_1/kernel/read*
T0* 
_output_shapes
:
ђђ
H
add_19Addmul_39mul_40* 
_output_shapes
:
ђђ*
T0
Х
	Assign_16Assigntarget/q2/dense_1/kerneladd_19*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
V
mul_41/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
Z
mul_41Mulmul_41/xtarget/q2/dense_1/bias/read*
T0*
_output_shapes	
:ђ
V
mul_42/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
X
mul_42Mulmul_42/xmain/q2/dense_1/bias/read*
T0*
_output_shapes	
:ђ
C
add_20Addmul_41mul_42*
T0*
_output_shapes	
:ђ
Г
	Assign_17Assigntarget/q2/dense_1/biasadd_20*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*)
_class
loc:@target/q2/dense_1/bias
V
mul_43/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
`
mul_43Mulmul_43/xtarget/q2/dense_2/kernel/read*
_output_shapes
:	ђ*
T0
V
mul_44/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
^
mul_44Mulmul_44/xmain/q2/dense_2/kernel/read*
T0*
_output_shapes
:	ђ
G
add_21Addmul_43mul_44*
T0*
_output_shapes
:	ђ
х
	Assign_18Assigntarget/q2/dense_2/kerneladd_21*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
V
mul_45/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *RИ~?
Y
mul_45Mulmul_45/xtarget/q2/dense_2/bias/read*
T0*
_output_shapes
:
V
mul_46/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
W
mul_46Mulmul_46/xmain/q2/dense_2/bias/read*
T0*
_output_shapes
:
B
add_22Addmul_45mul_46*
_output_shapes
:*
T0
г
	Assign_19Assigntarget/q2/dense_2/biasadd_22*
use_locking(*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
V
mul_47/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
]
mul_47Mulmul_47/xtarget/v/dense/kernel/read*
T0*
_output_shapes
:	Iђ
V
mul_48/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *
ОБ;
[
mul_48Mulmul_48/xmain/v/dense/kernel/read*
T0*
_output_shapes
:	Iђ
G
add_23Addmul_47mul_48*
T0*
_output_shapes
:	Iђ
»
	Assign_20Assigntarget/v/dense/kerneladd_23*
use_locking(*
T0*(
_class
loc:@target/v/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
V
mul_49/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
W
mul_49Mulmul_49/xtarget/v/dense/bias/read*
T0*
_output_shapes	
:ђ
V
mul_50/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *
ОБ;
U
mul_50Mulmul_50/xmain/v/dense/bias/read*
T0*
_output_shapes	
:ђ
C
add_24Addmul_49mul_50*
_output_shapes	
:ђ*
T0
Д
	Assign_21Assigntarget/v/dense/biasadd_24*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*&
_class
loc:@target/v/dense/bias
V
mul_51/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
`
mul_51Mulmul_51/xtarget/v/dense_1/kernel/read*
T0* 
_output_shapes
:
ђђ
V
mul_52/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *
ОБ;
^
mul_52Mulmul_52/xmain/v/dense_1/kernel/read*
T0* 
_output_shapes
:
ђђ
H
add_25Addmul_51mul_52* 
_output_shapes
:
ђђ*
T0
┤
	Assign_22Assigntarget/v/dense_1/kerneladd_25*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0**
_class 
loc:@target/v/dense_1/kernel
V
mul_53/xConst^Adam_1*
valueB
 *RИ~?*
dtype0*
_output_shapes
: 
Y
mul_53Mulmul_53/xtarget/v/dense_1/bias/read*
_output_shapes	
:ђ*
T0
V
mul_54/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
W
mul_54Mulmul_54/xmain/v/dense_1/bias/read*
T0*
_output_shapes	
:ђ
C
add_26Addmul_53mul_54*
_output_shapes	
:ђ*
T0
Ф
	Assign_23Assigntarget/v/dense_1/biasadd_26*
use_locking(*
T0*(
_class
loc:@target/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
V
mul_55/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *RИ~?
_
mul_55Mulmul_55/xtarget/v/dense_2/kernel/read*
T0*
_output_shapes
:	ђ
V
mul_56/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *
ОБ;
]
mul_56Mulmul_56/xmain/v/dense_2/kernel/read*
T0*
_output_shapes
:	ђ
G
add_27Addmul_55mul_56*
_output_shapes
:	ђ*
T0
│
	Assign_24Assigntarget/v/dense_2/kerneladd_27*
use_locking(*
T0**
_class 
loc:@target/v/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
V
mul_57/xConst^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *RИ~?
X
mul_57Mulmul_57/xtarget/v/dense_2/bias/read*
T0*
_output_shapes
:
V
mul_58/xConst^Adam_1*
valueB
 *
ОБ;*
dtype0*
_output_shapes
: 
V
mul_58Mulmul_58/xmain/v/dense_2/bias/read*
_output_shapes
:*
T0
B
add_28Addmul_57mul_58*
T0*
_output_shapes
:
ф
	Assign_25Assigntarget/v/dense_2/biasadd_28*
use_locking(*
T0*(
_class
loc:@target/v/dense_2/bias*
validate_shape(*
_output_shapes
:
К

group_depsNoOp^Adam_1^Assign	^Assign_1
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19	^Assign_2
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
─
	Assign_26Assigntarget/pi/dense/kernelmain/pi/dense/kernel/read*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
║
	Assign_27Assigntarget/pi/dense/biasmain/pi/dense/bias/read*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
╦
	Assign_28Assigntarget/pi/dense_1/kernelmain/pi/dense_1/kernel/read*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
└
	Assign_29Assigntarget/pi/dense_1/biasmain/pi/dense_1/bias/read*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias
╩
	Assign_30Assigntarget/pi/dense_2/kernelmain/pi/dense_2/kernel/read*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
┐
	Assign_31Assigntarget/pi/dense_2/biasmain/pi/dense_2/bias/read*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias
╩
	Assign_32Assigntarget/pi/dense_3/kernelmain/pi/dense_3/kernel/read*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_3/kernel
┐
	Assign_33Assigntarget/pi/dense_3/biasmain/pi/dense_3/bias/read*
use_locking(*
T0*)
_class
loc:@target/pi/dense_3/bias*
validate_shape(*
_output_shapes
:
─
	Assign_34Assigntarget/q1/dense/kernelmain/q1/dense/kernel/read*
T0*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
_output_shapes
:	hђ*
use_locking(
║
	Assign_35Assigntarget/q1/dense/biasmain/q1/dense/bias/read*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*'
_class
loc:@target/q1/dense/bias
╦
	Assign_36Assigntarget/q1/dense_1/kernelmain/q1/dense_1/kernel/read*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
└
	Assign_37Assigntarget/q1/dense_1/biasmain/q1/dense_1/bias/read*
use_locking(*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
╩
	Assign_38Assigntarget/q1/dense_2/kernelmain/q1/dense_2/kernel/read*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
┐
	Assign_39Assigntarget/q1/dense_2/biasmain/q1/dense_2/bias/read*
T0*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
─
	Assign_40Assigntarget/q2/dense/kernelmain/q2/dense/kernel/read*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
║
	Assign_41Assigntarget/q2/dense/biasmain/q2/dense/bias/read*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
╦
	Assign_42Assigntarget/q2/dense_1/kernelmain/q2/dense_1/kernel/read*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
└
	Assign_43Assigntarget/q2/dense_1/biasmain/q2/dense_1/bias/read*
T0*)
_class
loc:@target/q2/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
╩
	Assign_44Assigntarget/q2/dense_2/kernelmain/q2/dense_2/kernel/read*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
┐
	Assign_45Assigntarget/q2/dense_2/biasmain/q2/dense_2/bias/read*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
┴
	Assign_46Assigntarget/v/dense/kernelmain/v/dense/kernel/read*
use_locking(*
T0*(
_class
loc:@target/v/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
и
	Assign_47Assigntarget/v/dense/biasmain/v/dense/bias/read*
use_locking(*
T0*&
_class
loc:@target/v/dense/bias*
validate_shape(*
_output_shapes	
:ђ
╚
	Assign_48Assigntarget/v/dense_1/kernelmain/v/dense_1/kernel/read*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0**
_class 
loc:@target/v/dense_1/kernel
й
	Assign_49Assigntarget/v/dense_1/biasmain/v/dense_1/bias/read*
use_locking(*
T0*(
_class
loc:@target/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
К
	Assign_50Assigntarget/v/dense_2/kernelmain/v/dense_2/kernel/read*
use_locking(*
T0**
_class 
loc:@target/v/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
╝
	Assign_51Assigntarget/v/dense_2/biasmain/v/dense_2/bias/read*
use_locking(*
T0*(
_class
loc:@target/v/dense_2/bias*
validate_shape(*
_output_shapes
:
╠
group_deps_1NoOp
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
^Assign_51
љ
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^main/pi/dense/bias/Adam/Assign!^main/pi/dense/bias/Adam_1/Assign^main/pi/dense/bias/Assign!^main/pi/dense/kernel/Adam/Assign#^main/pi/dense/kernel/Adam_1/Assign^main/pi/dense/kernel/Assign!^main/pi/dense_1/bias/Adam/Assign#^main/pi/dense_1/bias/Adam_1/Assign^main/pi/dense_1/bias/Assign#^main/pi/dense_1/kernel/Adam/Assign%^main/pi/dense_1/kernel/Adam_1/Assign^main/pi/dense_1/kernel/Assign!^main/pi/dense_2/bias/Adam/Assign#^main/pi/dense_2/bias/Adam_1/Assign^main/pi/dense_2/bias/Assign#^main/pi/dense_2/kernel/Adam/Assign%^main/pi/dense_2/kernel/Adam_1/Assign^main/pi/dense_2/kernel/Assign!^main/pi/dense_3/bias/Adam/Assign#^main/pi/dense_3/bias/Adam_1/Assign^main/pi/dense_3/bias/Assign#^main/pi/dense_3/kernel/Adam/Assign%^main/pi/dense_3/kernel/Adam_1/Assign^main/pi/dense_3/kernel/Assign^main/q1/dense/bias/Adam/Assign!^main/q1/dense/bias/Adam_1/Assign^main/q1/dense/bias/Assign!^main/q1/dense/kernel/Adam/Assign#^main/q1/dense/kernel/Adam_1/Assign^main/q1/dense/kernel/Assign!^main/q1/dense_1/bias/Adam/Assign#^main/q1/dense_1/bias/Adam_1/Assign^main/q1/dense_1/bias/Assign#^main/q1/dense_1/kernel/Adam/Assign%^main/q1/dense_1/kernel/Adam_1/Assign^main/q1/dense_1/kernel/Assign!^main/q1/dense_2/bias/Adam/Assign#^main/q1/dense_2/bias/Adam_1/Assign^main/q1/dense_2/bias/Assign#^main/q1/dense_2/kernel/Adam/Assign%^main/q1/dense_2/kernel/Adam_1/Assign^main/q1/dense_2/kernel/Assign^main/q2/dense/bias/Adam/Assign!^main/q2/dense/bias/Adam_1/Assign^main/q2/dense/bias/Assign!^main/q2/dense/kernel/Adam/Assign#^main/q2/dense/kernel/Adam_1/Assign^main/q2/dense/kernel/Assign!^main/q2/dense_1/bias/Adam/Assign#^main/q2/dense_1/bias/Adam_1/Assign^main/q2/dense_1/bias/Assign#^main/q2/dense_1/kernel/Adam/Assign%^main/q2/dense_1/kernel/Adam_1/Assign^main/q2/dense_1/kernel/Assign!^main/q2/dense_2/bias/Adam/Assign#^main/q2/dense_2/bias/Adam_1/Assign^main/q2/dense_2/bias/Assign#^main/q2/dense_2/kernel/Adam/Assign%^main/q2/dense_2/kernel/Adam_1/Assign^main/q2/dense_2/kernel/Assign^main/v/dense/bias/Adam/Assign ^main/v/dense/bias/Adam_1/Assign^main/v/dense/bias/Assign ^main/v/dense/kernel/Adam/Assign"^main/v/dense/kernel/Adam_1/Assign^main/v/dense/kernel/Assign ^main/v/dense_1/bias/Adam/Assign"^main/v/dense_1/bias/Adam_1/Assign^main/v/dense_1/bias/Assign"^main/v/dense_1/kernel/Adam/Assign$^main/v/dense_1/kernel/Adam_1/Assign^main/v/dense_1/kernel/Assign ^main/v/dense_2/bias/Adam/Assign"^main/v/dense_2/bias/Adam_1/Assign^main/v/dense_2/bias/Assign"^main/v/dense_2/kernel/Adam/Assign$^main/v/dense_2/kernel/Adam_1/Assign^main/v/dense_2/kernel/Assign^target/pi/dense/bias/Assign^target/pi/dense/kernel/Assign^target/pi/dense_1/bias/Assign ^target/pi/dense_1/kernel/Assign^target/pi/dense_2/bias/Assign ^target/pi/dense_2/kernel/Assign^target/pi/dense_3/bias/Assign ^target/pi/dense_3/kernel/Assign^target/q1/dense/bias/Assign^target/q1/dense/kernel/Assign^target/q1/dense_1/bias/Assign ^target/q1/dense_1/kernel/Assign^target/q1/dense_2/bias/Assign ^target/q1/dense_2/kernel/Assign^target/q2/dense/bias/Assign^target/q2/dense/kernel/Assign^target/q2/dense_1/bias/Assign ^target/q2/dense_1/kernel/Assign^target/q2/dense_2/bias/Assign ^target/q2/dense_2/kernel/Assign^target/v/dense/bias/Assign^target/v/dense/kernel/Assign^target/v/dense_1/bias/Assign^target/v/dense_1/kernel/Assign^target/v/dense_2/bias/Assign^target/v/dense_2/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
ё
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_ddf8c13766b34536a2a0fb0107d7c0e8/part*
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
є
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:l*╣
value»BгlBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/pi/dense_3/biasBmain/pi/dense_3/bias/AdamBmain/pi/dense_3/bias/Adam_1Bmain/pi/dense_3/kernelBmain/pi/dense_3/kernel/AdamBmain/pi/dense_3/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Bmain/v/dense/biasBmain/v/dense/bias/AdamBmain/v/dense/bias/Adam_1Bmain/v/dense/kernelBmain/v/dense/kernel/AdamBmain/v/dense/kernel/Adam_1Bmain/v/dense_1/biasBmain/v/dense_1/bias/AdamBmain/v/dense_1/bias/Adam_1Bmain/v/dense_1/kernelBmain/v/dense_1/kernel/AdamBmain/v/dense_1/kernel/Adam_1Bmain/v/dense_2/biasBmain/v/dense_2/bias/AdamBmain/v/dense_2/bias/Adam_1Bmain/v/dense_2/kernelBmain/v/dense_2/kernel/AdamBmain/v/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/pi/dense_3/biasBtarget/pi/dense_3/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernelBtarget/v/dense/biasBtarget/v/dense/kernelBtarget/v/dense_1/biasBtarget/v/dense_1/kernelBtarget/v/dense_2/biasBtarget/v/dense_2/kernel
Й
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:l*ь
valueсBЯlB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ѓ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/pi/dense_3/biasmain/pi/dense_3/bias/Adammain/pi/dense_3/bias/Adam_1main/pi/dense_3/kernelmain/pi/dense_3/kernel/Adammain/pi/dense_3/kernel/Adam_1main/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1main/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1main/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1main/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1main/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1main/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1main/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1main/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1main/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1main/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1main/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1main/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1main/v/dense/biasmain/v/dense/bias/Adammain/v/dense/bias/Adam_1main/v/dense/kernelmain/v/dense/kernel/Adammain/v/dense/kernel/Adam_1main/v/dense_1/biasmain/v/dense_1/bias/Adammain/v/dense_1/bias/Adam_1main/v/dense_1/kernelmain/v/dense_1/kernel/Adammain/v/dense_1/kernel/Adam_1main/v/dense_2/biasmain/v/dense_2/bias/Adammain/v/dense_2/bias/Adam_1main/v/dense_2/kernelmain/v/dense_2/kernel/Adammain/v/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/pi/dense_3/biastarget/pi/dense_3/kerneltarget/q1/dense/biastarget/q1/dense/kerneltarget/q1/dense_1/biastarget/q1/dense_1/kerneltarget/q1/dense_2/biastarget/q1/dense_2/kerneltarget/q2/dense/biastarget/q2/dense/kerneltarget/q2/dense_1/biastarget/q2/dense_1/kerneltarget/q2/dense_2/biastarget/q2/dense_2/kerneltarget/v/dense/biastarget/v/dense/kerneltarget/v/dense_1/biastarget/v/dense_1/kerneltarget/v/dense_2/biastarget/v/dense_2/kernel*z
dtypesp
n2l
Љ
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ю
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
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
Ѕ
save/RestoreV2/tensor_namesConst*╣
value»BгlBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/pi/dense_3/biasBmain/pi/dense_3/bias/AdamBmain/pi/dense_3/bias/Adam_1Bmain/pi/dense_3/kernelBmain/pi/dense_3/kernel/AdamBmain/pi/dense_3/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Bmain/v/dense/biasBmain/v/dense/bias/AdamBmain/v/dense/bias/Adam_1Bmain/v/dense/kernelBmain/v/dense/kernel/AdamBmain/v/dense/kernel/Adam_1Bmain/v/dense_1/biasBmain/v/dense_1/bias/AdamBmain/v/dense_1/bias/Adam_1Bmain/v/dense_1/kernelBmain/v/dense_1/kernel/AdamBmain/v/dense_1/kernel/Adam_1Bmain/v/dense_2/biasBmain/v/dense_2/bias/AdamBmain/v/dense_2/bias/Adam_1Bmain/v/dense_2/kernelBmain/v/dense_2/kernel/AdamBmain/v/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/pi/dense_3/biasBtarget/pi/dense_3/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernelBtarget/v/dense/biasBtarget/v/dense/kernelBtarget/v/dense_1/biasBtarget/v/dense_1/kernelBtarget/v/dense_2/biasBtarget/v/dense_2/kernel*
dtype0*
_output_shapes
:l
┴
save/RestoreV2/shape_and_slicesConst*ь
valueсBЯlB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:l
ф
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*к
_output_shapes│
░::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*z
dtypesp
n2l
Б
save/AssignAssignbeta1_powersave/RestoreV2*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Е
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Д
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
Е
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: 
│
save/Assign_4Assignmain/pi/dense/biassave/RestoreV2:4*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
И
save/Assign_5Assignmain/pi/dense/bias/Adamsave/RestoreV2:5*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ
║
save/Assign_6Assignmain/pi/dense/bias/Adam_1save/RestoreV2:6*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
╗
save/Assign_7Assignmain/pi/dense/kernelsave/RestoreV2:7*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
└
save/Assign_8Assignmain/pi/dense/kernel/Adamsave/RestoreV2:8*
validate_shape(*
_output_shapes
:	Iђ*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel
┬
save/Assign_9Assignmain/pi/dense/kernel/Adam_1save/RestoreV2:9*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
╣
save/Assign_10Assignmain/pi/dense_1/biassave/RestoreV2:10*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
Й
save/Assign_11Assignmain/pi/dense_1/bias/Adamsave/RestoreV2:11*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
└
save/Assign_12Assignmain/pi/dense_1/bias/Adam_1save/RestoreV2:12*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias
┬
save/Assign_13Assignmain/pi/dense_1/kernelsave/RestoreV2:13*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
К
save/Assign_14Assignmain/pi/dense_1/kernel/Adamsave/RestoreV2:14*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
╔
save/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save/RestoreV2:15*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
И
save/Assign_16Assignmain/pi/dense_2/biassave/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias
й
save/Assign_17Assignmain/pi/dense_2/bias/Adamsave/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias
┐
save/Assign_18Assignmain/pi/dense_2/bias/Adam_1save/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
┴
save/Assign_19Assignmain/pi/dense_2/kernelsave/RestoreV2:19*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
к
save/Assign_20Assignmain/pi/dense_2/kernel/Adamsave/RestoreV2:20*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
╚
save/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save/RestoreV2:21*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
И
save/Assign_22Assignmain/pi/dense_3/biassave/RestoreV2:22*
T0*'
_class
loc:@main/pi/dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
й
save/Assign_23Assignmain/pi/dense_3/bias/Adamsave/RestoreV2:23*
T0*'
_class
loc:@main/pi/dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
┐
save/Assign_24Assignmain/pi/dense_3/bias/Adam_1save/RestoreV2:24*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_3/bias
┴
save/Assign_25Assignmain/pi/dense_3/kernelsave/RestoreV2:25*
use_locking(*
T0*)
_class
loc:@main/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ
к
save/Assign_26Assignmain/pi/dense_3/kernel/Adamsave/RestoreV2:26*
use_locking(*
T0*)
_class
loc:@main/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ
╚
save/Assign_27Assignmain/pi/dense_3/kernel/Adam_1save/RestoreV2:27*
T0*)
_class
loc:@main/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
х
save/Assign_28Assignmain/q1/dense/biassave/RestoreV2:28*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ
║
save/Assign_29Assignmain/q1/dense/bias/Adamsave/RestoreV2:29*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ
╝
save/Assign_30Assignmain/q1/dense/bias/Adam_1save/RestoreV2:30*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ
й
save/Assign_31Assignmain/q1/dense/kernelsave/RestoreV2:31*
validate_shape(*
_output_shapes
:	hђ*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel
┬
save/Assign_32Assignmain/q1/dense/kernel/Adamsave/RestoreV2:32*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
─
save/Assign_33Assignmain/q1/dense/kernel/Adam_1save/RestoreV2:33*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
╣
save/Assign_34Assignmain/q1/dense_1/biassave/RestoreV2:34*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
Й
save/Assign_35Assignmain/q1/dense_1/bias/Adamsave/RestoreV2:35*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
└
save/Assign_36Assignmain/q1/dense_1/bias/Adam_1save/RestoreV2:36*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
┬
save/Assign_37Assignmain/q1/dense_1/kernelsave/RestoreV2:37*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
К
save/Assign_38Assignmain/q1/dense_1/kernel/Adamsave/RestoreV2:38*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
╔
save/Assign_39Assignmain/q1/dense_1/kernel/Adam_1save/RestoreV2:39*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
И
save/Assign_40Assignmain/q1/dense_2/biassave/RestoreV2:40*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
й
save/Assign_41Assignmain/q1/dense_2/bias/Adamsave/RestoreV2:41*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias
┐
save/Assign_42Assignmain/q1/dense_2/bias/Adam_1save/RestoreV2:42*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
┴
save/Assign_43Assignmain/q1/dense_2/kernelsave/RestoreV2:43*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
к
save/Assign_44Assignmain/q1/dense_2/kernel/Adamsave/RestoreV2:44*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel
╚
save/Assign_45Assignmain/q1/dense_2/kernel/Adam_1save/RestoreV2:45*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel
х
save/Assign_46Assignmain/q2/dense/biassave/RestoreV2:46*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes	
:ђ
║
save/Assign_47Assignmain/q2/dense/bias/Adamsave/RestoreV2:47*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
╝
save/Assign_48Assignmain/q2/dense/bias/Adam_1save/RestoreV2:48*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes	
:ђ
й
save/Assign_49Assignmain/q2/dense/kernelsave/RestoreV2:49*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
┬
save/Assign_50Assignmain/q2/dense/kernel/Adamsave/RestoreV2:50*
validate_shape(*
_output_shapes
:	hђ*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel
─
save/Assign_51Assignmain/q2/dense/kernel/Adam_1save/RestoreV2:51*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
╣
save/Assign_52Assignmain/q2/dense_1/biassave/RestoreV2:52*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
Й
save/Assign_53Assignmain/q2/dense_1/bias/Adamsave/RestoreV2:53*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
└
save/Assign_54Assignmain/q2/dense_1/bias/Adam_1save/RestoreV2:54*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
┬
save/Assign_55Assignmain/q2/dense_1/kernelsave/RestoreV2:55*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel
К
save/Assign_56Assignmain/q2/dense_1/kernel/Adamsave/RestoreV2:56*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
╔
save/Assign_57Assignmain/q2/dense_1/kernel/Adam_1save/RestoreV2:57*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
И
save/Assign_58Assignmain/q2/dense_2/biassave/RestoreV2:58*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
й
save/Assign_59Assignmain/q2/dense_2/bias/Adamsave/RestoreV2:59*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias
┐
save/Assign_60Assignmain/q2/dense_2/bias/Adam_1save/RestoreV2:60*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias
┴
save/Assign_61Assignmain/q2/dense_2/kernelsave/RestoreV2:61*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
к
save/Assign_62Assignmain/q2/dense_2/kernel/Adamsave/RestoreV2:62*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
╚
save/Assign_63Assignmain/q2/dense_2/kernel/Adam_1save/RestoreV2:63*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel
│
save/Assign_64Assignmain/v/dense/biassave/RestoreV2:64*
use_locking(*
T0*$
_class
loc:@main/v/dense/bias*
validate_shape(*
_output_shapes	
:ђ
И
save/Assign_65Assignmain/v/dense/bias/Adamsave/RestoreV2:65*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*$
_class
loc:@main/v/dense/bias
║
save/Assign_66Assignmain/v/dense/bias/Adam_1save/RestoreV2:66*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*$
_class
loc:@main/v/dense/bias
╗
save/Assign_67Assignmain/v/dense/kernelsave/RestoreV2:67*
validate_shape(*
_output_shapes
:	Iђ*
use_locking(*
T0*&
_class
loc:@main/v/dense/kernel
└
save/Assign_68Assignmain/v/dense/kernel/Adamsave/RestoreV2:68*
validate_shape(*
_output_shapes
:	Iђ*
use_locking(*
T0*&
_class
loc:@main/v/dense/kernel
┬
save/Assign_69Assignmain/v/dense/kernel/Adam_1save/RestoreV2:69*
T0*&
_class
loc:@main/v/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ*
use_locking(
и
save/Assign_70Assignmain/v/dense_1/biassave/RestoreV2:70*
use_locking(*
T0*&
_class
loc:@main/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
╝
save/Assign_71Assignmain/v/dense_1/bias/Adamsave/RestoreV2:71*
use_locking(*
T0*&
_class
loc:@main/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
Й
save/Assign_72Assignmain/v/dense_1/bias/Adam_1save/RestoreV2:72*
T0*&
_class
loc:@main/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
└
save/Assign_73Assignmain/v/dense_1/kernelsave/RestoreV2:73*
use_locking(*
T0*(
_class
loc:@main/v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
┼
save/Assign_74Assignmain/v/dense_1/kernel/Adamsave/RestoreV2:74*
use_locking(*
T0*(
_class
loc:@main/v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
К
save/Assign_75Assignmain/v/dense_1/kernel/Adam_1save/RestoreV2:75*
T0*(
_class
loc:@main/v/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
Х
save/Assign_76Assignmain/v/dense_2/biassave/RestoreV2:76*
T0*&
_class
loc:@main/v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╗
save/Assign_77Assignmain/v/dense_2/bias/Adamsave/RestoreV2:77*
use_locking(*
T0*&
_class
loc:@main/v/dense_2/bias*
validate_shape(*
_output_shapes
:
й
save/Assign_78Assignmain/v/dense_2/bias/Adam_1save/RestoreV2:78*
use_locking(*
T0*&
_class
loc:@main/v/dense_2/bias*
validate_shape(*
_output_shapes
:
┐
save/Assign_79Assignmain/v/dense_2/kernelsave/RestoreV2:79*
use_locking(*
T0*(
_class
loc:@main/v/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
─
save/Assign_80Assignmain/v/dense_2/kernel/Adamsave/RestoreV2:80*
use_locking(*
T0*(
_class
loc:@main/v/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
к
save/Assign_81Assignmain/v/dense_2/kernel/Adam_1save/RestoreV2:81*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*(
_class
loc:@main/v/dense_2/kernel
╣
save/Assign_82Assigntarget/pi/dense/biassave/RestoreV2:82*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
┴
save/Assign_83Assigntarget/pi/dense/kernelsave/RestoreV2:83*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ*
use_locking(
й
save/Assign_84Assigntarget/pi/dense_1/biassave/RestoreV2:84*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
к
save/Assign_85Assigntarget/pi/dense_1/kernelsave/RestoreV2:85*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
╝
save/Assign_86Assigntarget/pi/dense_2/biassave/RestoreV2:86*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
┼
save/Assign_87Assigntarget/pi/dense_2/kernelsave/RestoreV2:87*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
╝
save/Assign_88Assigntarget/pi/dense_3/biassave/RestoreV2:88*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/pi/dense_3/bias
┼
save/Assign_89Assigntarget/pi/dense_3/kernelsave/RestoreV2:89*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_3/kernel*
validate_shape(*
_output_shapes
:	ђ
╣
save/Assign_90Assigntarget/q1/dense/biassave/RestoreV2:90*
T0*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
┴
save/Assign_91Assigntarget/q1/dense/kernelsave/RestoreV2:91*
use_locking(*
T0*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
й
save/Assign_92Assigntarget/q1/dense_1/biassave/RestoreV2:92*
use_locking(*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ
к
save/Assign_93Assigntarget/q1/dense_1/kernelsave/RestoreV2:93*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(* 
_output_shapes
:
ђђ
╝
save/Assign_94Assigntarget/q1/dense_2/biassave/RestoreV2:94*
use_locking(*
T0*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
┼
save/Assign_95Assigntarget/q1/dense_2/kernelsave/RestoreV2:95*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ
╣
save/Assign_96Assigntarget/q2/dense/biassave/RestoreV2:96*
use_locking(*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes	
:ђ
┴
save/Assign_97Assigntarget/q2/dense/kernelsave/RestoreV2:97*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes
:	hђ
й
save/Assign_98Assigntarget/q2/dense_1/biassave/RestoreV2:98*
T0*)
_class
loc:@target/q2/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
к
save/Assign_99Assigntarget/q2/dense_1/kernelsave/RestoreV2:99*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_1/kernel
Й
save/Assign_100Assigntarget/q2/dense_2/biassave/RestoreV2:100*
use_locking(*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
К
save/Assign_101Assigntarget/q2/dense_2/kernelsave/RestoreV2:101*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
╣
save/Assign_102Assigntarget/v/dense/biassave/RestoreV2:102*
T0*&
_class
loc:@target/v/dense/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
┴
save/Assign_103Assigntarget/v/dense/kernelsave/RestoreV2:103*
use_locking(*
T0*(
_class
loc:@target/v/dense/kernel*
validate_shape(*
_output_shapes
:	Iђ
й
save/Assign_104Assigntarget/v/dense_1/biassave/RestoreV2:104*
T0*(
_class
loc:@target/v/dense_1/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
к
save/Assign_105Assigntarget/v/dense_1/kernelsave/RestoreV2:105*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0**
_class 
loc:@target/v/dense_1/kernel
╝
save/Assign_106Assigntarget/v/dense_2/biassave/RestoreV2:106*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*(
_class
loc:@target/v/dense_2/bias
┼
save/Assign_107Assigntarget/v/dense_2/kernelsave/RestoreV2:107*
T0**
_class 
loc:@target/v/dense_2/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
┬
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"ў8
trainable_variablesђ8§7
Є
main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:021main/pi/dense/kernel/Initializer/random_uniform:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08
Ј
main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:023main/pi/dense_1/kernel/Initializer/random_uniform:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08
Ј
main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08
Ј
main/pi/dense_3/kernel:0main/pi/dense_3/kernel/Assignmain/pi/dense_3/kernel/read:023main/pi/dense_3/kernel/Initializer/random_uniform:08
~
main/pi/dense_3/bias:0main/pi/dense_3/bias/Assignmain/pi/dense_3/bias/read:02(main/pi/dense_3/bias/Initializer/zeros:08
Є
main/q1/dense/kernel:0main/q1/dense/kernel/Assignmain/q1/dense/kernel/read:021main/q1/dense/kernel/Initializer/random_uniform:08
v
main/q1/dense/bias:0main/q1/dense/bias/Assignmain/q1/dense/bias/read:02&main/q1/dense/bias/Initializer/zeros:08
Ј
main/q1/dense_1/kernel:0main/q1/dense_1/kernel/Assignmain/q1/dense_1/kernel/read:023main/q1/dense_1/kernel/Initializer/random_uniform:08
~
main/q1/dense_1/bias:0main/q1/dense_1/bias/Assignmain/q1/dense_1/bias/read:02(main/q1/dense_1/bias/Initializer/zeros:08
Ј
main/q1/dense_2/kernel:0main/q1/dense_2/kernel/Assignmain/q1/dense_2/kernel/read:023main/q1/dense_2/kernel/Initializer/random_uniform:08
~
main/q1/dense_2/bias:0main/q1/dense_2/bias/Assignmain/q1/dense_2/bias/read:02(main/q1/dense_2/bias/Initializer/zeros:08
Є
main/q2/dense/kernel:0main/q2/dense/kernel/Assignmain/q2/dense/kernel/read:021main/q2/dense/kernel/Initializer/random_uniform:08
v
main/q2/dense/bias:0main/q2/dense/bias/Assignmain/q2/dense/bias/read:02&main/q2/dense/bias/Initializer/zeros:08
Ј
main/q2/dense_1/kernel:0main/q2/dense_1/kernel/Assignmain/q2/dense_1/kernel/read:023main/q2/dense_1/kernel/Initializer/random_uniform:08
~
main/q2/dense_1/bias:0main/q2/dense_1/bias/Assignmain/q2/dense_1/bias/read:02(main/q2/dense_1/bias/Initializer/zeros:08
Ј
main/q2/dense_2/kernel:0main/q2/dense_2/kernel/Assignmain/q2/dense_2/kernel/read:023main/q2/dense_2/kernel/Initializer/random_uniform:08
~
main/q2/dense_2/bias:0main/q2/dense_2/bias/Assignmain/q2/dense_2/bias/read:02(main/q2/dense_2/bias/Initializer/zeros:08
Ѓ
main/v/dense/kernel:0main/v/dense/kernel/Assignmain/v/dense/kernel/read:020main/v/dense/kernel/Initializer/random_uniform:08
r
main/v/dense/bias:0main/v/dense/bias/Assignmain/v/dense/bias/read:02%main/v/dense/bias/Initializer/zeros:08
І
main/v/dense_1/kernel:0main/v/dense_1/kernel/Assignmain/v/dense_1/kernel/read:022main/v/dense_1/kernel/Initializer/random_uniform:08
z
main/v/dense_1/bias:0main/v/dense_1/bias/Assignmain/v/dense_1/bias/read:02'main/v/dense_1/bias/Initializer/zeros:08
І
main/v/dense_2/kernel:0main/v/dense_2/kernel/Assignmain/v/dense_2/kernel/read:022main/v/dense_2/kernel/Initializer/random_uniform:08
z
main/v/dense_2/bias:0main/v/dense_2/bias/Assignmain/v/dense_2/bias/read:02'main/v/dense_2/bias/Initializer/zeros:08
Ј
target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:023target/pi/dense/kernel/Initializer/random_uniform:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08
Ќ
target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:025target/pi/dense_1/kernel/Initializer/random_uniform:08
є
target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08
Ќ
target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08
є
target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08
Ќ
target/pi/dense_3/kernel:0target/pi/dense_3/kernel/Assigntarget/pi/dense_3/kernel/read:025target/pi/dense_3/kernel/Initializer/random_uniform:08
є
target/pi/dense_3/bias:0target/pi/dense_3/bias/Assigntarget/pi/dense_3/bias/read:02*target/pi/dense_3/bias/Initializer/zeros:08
Ј
target/q1/dense/kernel:0target/q1/dense/kernel/Assigntarget/q1/dense/kernel/read:023target/q1/dense/kernel/Initializer/random_uniform:08
~
target/q1/dense/bias:0target/q1/dense/bias/Assigntarget/q1/dense/bias/read:02(target/q1/dense/bias/Initializer/zeros:08
Ќ
target/q1/dense_1/kernel:0target/q1/dense_1/kernel/Assigntarget/q1/dense_1/kernel/read:025target/q1/dense_1/kernel/Initializer/random_uniform:08
є
target/q1/dense_1/bias:0target/q1/dense_1/bias/Assigntarget/q1/dense_1/bias/read:02*target/q1/dense_1/bias/Initializer/zeros:08
Ќ
target/q1/dense_2/kernel:0target/q1/dense_2/kernel/Assigntarget/q1/dense_2/kernel/read:025target/q1/dense_2/kernel/Initializer/random_uniform:08
є
target/q1/dense_2/bias:0target/q1/dense_2/bias/Assigntarget/q1/dense_2/bias/read:02*target/q1/dense_2/bias/Initializer/zeros:08
Ј
target/q2/dense/kernel:0target/q2/dense/kernel/Assigntarget/q2/dense/kernel/read:023target/q2/dense/kernel/Initializer/random_uniform:08
~
target/q2/dense/bias:0target/q2/dense/bias/Assigntarget/q2/dense/bias/read:02(target/q2/dense/bias/Initializer/zeros:08
Ќ
target/q2/dense_1/kernel:0target/q2/dense_1/kernel/Assigntarget/q2/dense_1/kernel/read:025target/q2/dense_1/kernel/Initializer/random_uniform:08
є
target/q2/dense_1/bias:0target/q2/dense_1/bias/Assigntarget/q2/dense_1/bias/read:02*target/q2/dense_1/bias/Initializer/zeros:08
Ќ
target/q2/dense_2/kernel:0target/q2/dense_2/kernel/Assigntarget/q2/dense_2/kernel/read:025target/q2/dense_2/kernel/Initializer/random_uniform:08
є
target/q2/dense_2/bias:0target/q2/dense_2/bias/Assigntarget/q2/dense_2/bias/read:02*target/q2/dense_2/bias/Initializer/zeros:08
І
target/v/dense/kernel:0target/v/dense/kernel/Assigntarget/v/dense/kernel/read:022target/v/dense/kernel/Initializer/random_uniform:08
z
target/v/dense/bias:0target/v/dense/bias/Assigntarget/v/dense/bias/read:02'target/v/dense/bias/Initializer/zeros:08
Њ
target/v/dense_1/kernel:0target/v/dense_1/kernel/Assigntarget/v/dense_1/kernel/read:024target/v/dense_1/kernel/Initializer/random_uniform:08
ѓ
target/v/dense_1/bias:0target/v/dense_1/bias/Assigntarget/v/dense_1/bias/read:02)target/v/dense_1/bias/Initializer/zeros:08
Њ
target/v/dense_2/kernel:0target/v/dense_2/kernel/Assigntarget/v/dense_2/kernel/read:024target/v/dense_2/kernel/Initializer/random_uniform:08
ѓ
target/v/dense_2/bias:0target/v/dense_2/bias/Assigntarget/v/dense_2/bias/read:02)target/v/dense_2/bias/Initializer/zeros:08"╩J
cond_context╣JХJ
ы
main/pi/cond/cond_textmain/pi/cond/pred_id:0main/pi/cond/switch_t:0 *▀
main/pi/cond/Equal/Switch:1
main/pi/cond/Equal/y:0
main/pi/cond/Equal:0
main/pi/cond/Exp/Switch:1
main/pi/cond/Exp:0
main/pi/cond/Maximum/y:0
main/pi/cond/Maximum:0
$main/pi/cond/Sum/reduction_indices:0
main/pi/cond/Sum:0
main/pi/cond/cond/Log/Switch:1
main/pi/cond/cond/Log:0
main/pi/cond/cond/Merge:0
main/pi/cond/cond/Merge:1
main/pi/cond/cond/Pow/Switch:0
main/pi/cond/cond/Pow:0
main/pi/cond/cond/Switch:0
main/pi/cond/cond/Switch:1
main/pi/cond/cond/pred_id:0
main/pi/cond/cond/sub/Switch:0
main/pi/cond/cond/sub/x:0
main/pi/cond/cond/sub:0
main/pi/cond/cond/sub_1/y:0
main/pi/cond/cond/sub_1:0
main/pi/cond/cond/sub_2/x:0
main/pi/cond/cond/sub_2:0
main/pi/cond/cond/switch_f:0
main/pi/cond/cond/switch_t:0
main/pi/cond/cond/truediv:0
main/pi/cond/pred_id:0
main/pi/cond/switch_t:0
main/pi/sub:0
main/pi/sub_5:00
main/pi/cond/pred_id:0main/pi/cond/pred_id:0,
main/pi/sub_5:0main/pi/cond/Exp/Switch:1,
main/pi/sub:0main/pi/cond/Equal/Switch:12Р
▀
main/pi/cond/cond/cond_textmain/pi/cond/cond/pred_id:0main/pi/cond/cond/switch_t:0 *ѓ
main/pi/cond/Maximum:0
main/pi/cond/cond/Log/Switch:1
main/pi/cond/cond/Log:0
main/pi/cond/cond/pred_id:0
main/pi/cond/cond/switch_t:08
main/pi/cond/Maximum:0main/pi/cond/cond/Log/Switch:1:
main/pi/cond/cond/pred_id:0main/pi/cond/cond/pred_id:02▄
┘
main/pi/cond/cond/cond_text_1main/pi/cond/cond/pred_id:0main/pi/cond/cond/switch_f:0*Ч
main/pi/cond/Equal/Switch:1
main/pi/cond/Maximum:0
main/pi/cond/cond/Pow/Switch:0
main/pi/cond/cond/Pow:0
main/pi/cond/cond/pred_id:0
main/pi/cond/cond/sub/Switch:0
main/pi/cond/cond/sub/x:0
main/pi/cond/cond/sub:0
main/pi/cond/cond/sub_1/y:0
main/pi/cond/cond/sub_1:0
main/pi/cond/cond/sub_2/x:0
main/pi/cond/cond/sub_2:0
main/pi/cond/cond/switch_f:0
main/pi/cond/cond/truediv:0
main/pi/sub:0/
main/pi/sub:0main/pi/cond/cond/sub/Switch:0:
main/pi/cond/Equal/Switch:1main/pi/cond/Equal/Switch:18
main/pi/cond/Maximum:0main/pi/cond/cond/Pow/Switch:0:
main/pi/cond/cond/pred_id:0main/pi/cond/cond/pred_id:0
 
main/pi/cond/cond_text_1main/pi/cond/pred_id:0main/pi/cond/switch_f:0*▒	
main/pi/cond/Equal_1/y:0
main/pi/cond/Equal_1:0
main/pi/cond/Exp_1/Switch:0
main/pi/cond/Exp_1:0
main/pi/cond/Maximum_1/y:0
main/pi/cond/Maximum_1:0
main/pi/cond/Minimum:0
main/pi/cond/Pow/x:0
main/pi/cond/Pow:0
&main/pi/cond/Sum_1/reduction_indices:0
main/pi/cond/Sum_1:0
 main/pi/cond/cond_1/Log/Switch:1
main/pi/cond/cond_1/Log:0
main/pi/cond/cond_1/Merge:0
main/pi/cond/cond_1/Merge:1
 main/pi/cond/cond_1/Pow/Switch:0
main/pi/cond/cond_1/Pow:0
main/pi/cond/cond_1/Switch:0
main/pi/cond/cond_1/Switch:1
main/pi/cond/cond_1/pred_id:0
 main/pi/cond/cond_1/sub/Switch:0
main/pi/cond/cond_1/sub/x:0
main/pi/cond/cond_1/sub:0
main/pi/cond/cond_1/sub_1/y:0
main/pi/cond/cond_1/sub_1:0
main/pi/cond/cond_1/sub_2/x:0
main/pi/cond/cond_1/sub_2:0
main/pi/cond/cond_1/switch_f:0
main/pi/cond/cond_1/switch_t:0
main/pi/cond/cond_1/truediv:0
main/pi/cond/pred_id:0
main/pi/cond/sub/Switch:0
main/pi/cond/sub/x:0
main/pi/cond/sub:0
main/pi/cond/switch_f:0
main/pi/cond/truediv/x:0
main/pi/cond/truediv:0
main/pi/sub:0
main/pi/sub_5:00
main/pi/cond/pred_id:0main/pi/cond/pred_id:0.
main/pi/sub_5:0main/pi/cond/Exp_1/Switch:0*
main/pi/sub:0main/pi/cond/sub/Switch:02Щ
э
main/pi/cond/cond_1/cond_textmain/pi/cond/cond_1/pred_id:0main/pi/cond/cond_1/switch_t:0 *ћ
main/pi/cond/Maximum_1:0
 main/pi/cond/cond_1/Log/Switch:1
main/pi/cond/cond_1/Log:0
main/pi/cond/cond_1/pred_id:0
main/pi/cond/cond_1/switch_t:0<
main/pi/cond/Maximum_1:0 main/pi/cond/cond_1/Log/Switch:1>
main/pi/cond/cond_1/pred_id:0main/pi/cond/cond_1/pred_id:02ђ
§
main/pi/cond/cond_1/cond_text_1main/pi/cond/cond_1/pred_id:0main/pi/cond/cond_1/switch_f:0*џ
main/pi/cond/Maximum_1:0
 main/pi/cond/cond_1/Pow/Switch:0
main/pi/cond/cond_1/Pow:0
main/pi/cond/cond_1/pred_id:0
 main/pi/cond/cond_1/sub/Switch:0
main/pi/cond/cond_1/sub/x:0
main/pi/cond/cond_1/sub:0
main/pi/cond/cond_1/sub_1/y:0
main/pi/cond/cond_1/sub_1:0
main/pi/cond/cond_1/sub_2/x:0
main/pi/cond/cond_1/sub_2:0
main/pi/cond/cond_1/switch_f:0
main/pi/cond/cond_1/truediv:0
main/pi/cond/sub/Switch:0
main/pi/sub:06
main/pi/cond/sub/Switch:0main/pi/cond/sub/Switch:01
main/pi/sub:0 main/pi/cond/cond_1/sub/Switch:0<
main/pi/cond/Maximum_1:0 main/pi/cond/cond_1/Pow/Switch:0>
main/pi/cond/cond_1/pred_id:0main/pi/cond/cond_1/pred_id:0
Ј
target/pi/cond/cond_texttarget/pi/cond/pred_id:0target/pi/cond/switch_t:0 *Ф
target/pi/cond/Equal/Switch:1
target/pi/cond/Equal/y:0
target/pi/cond/Equal:0
target/pi/cond/Exp/Switch:1
target/pi/cond/Exp:0
target/pi/cond/Maximum/y:0
target/pi/cond/Maximum:0
&target/pi/cond/Sum/reduction_indices:0
target/pi/cond/Sum:0
 target/pi/cond/cond/Log/Switch:1
target/pi/cond/cond/Log:0
target/pi/cond/cond/Merge:0
target/pi/cond/cond/Merge:1
 target/pi/cond/cond/Pow/Switch:0
target/pi/cond/cond/Pow:0
target/pi/cond/cond/Switch:0
target/pi/cond/cond/Switch:1
target/pi/cond/cond/pred_id:0
 target/pi/cond/cond/sub/Switch:0
target/pi/cond/cond/sub/x:0
target/pi/cond/cond/sub:0
target/pi/cond/cond/sub_1/y:0
target/pi/cond/cond/sub_1:0
target/pi/cond/cond/sub_2/x:0
target/pi/cond/cond/sub_2:0
target/pi/cond/cond/switch_f:0
target/pi/cond/cond/switch_t:0
target/pi/cond/cond/truediv:0
target/pi/cond/pred_id:0
target/pi/cond/switch_t:0
target/pi/sub:0
target/pi/sub_5:00
target/pi/sub_5:0target/pi/cond/Exp/Switch:14
target/pi/cond/pred_id:0target/pi/cond/pred_id:00
target/pi/sub:0target/pi/cond/Equal/Switch:12Щ
э
target/pi/cond/cond/cond_texttarget/pi/cond/cond/pred_id:0target/pi/cond/cond/switch_t:0 *ћ
target/pi/cond/Maximum:0
 target/pi/cond/cond/Log/Switch:1
target/pi/cond/cond/Log:0
target/pi/cond/cond/pred_id:0
target/pi/cond/cond/switch_t:0>
target/pi/cond/cond/pred_id:0target/pi/cond/cond/pred_id:0<
target/pi/cond/Maximum:0 target/pi/cond/cond/Log/Switch:12љ
Ї
target/pi/cond/cond/cond_text_1target/pi/cond/cond/pred_id:0target/pi/cond/cond/switch_f:0*ф
target/pi/cond/Equal/Switch:1
target/pi/cond/Maximum:0
 target/pi/cond/cond/Pow/Switch:0
target/pi/cond/cond/Pow:0
target/pi/cond/cond/pred_id:0
 target/pi/cond/cond/sub/Switch:0
target/pi/cond/cond/sub/x:0
target/pi/cond/cond/sub:0
target/pi/cond/cond/sub_1/y:0
target/pi/cond/cond/sub_1:0
target/pi/cond/cond/sub_2/x:0
target/pi/cond/cond/sub_2:0
target/pi/cond/cond/switch_f:0
target/pi/cond/cond/truediv:0
target/pi/sub:0>
target/pi/cond/cond/pred_id:0target/pi/cond/cond/pred_id:0<
target/pi/cond/Maximum:0 target/pi/cond/cond/Pow/Switch:0>
target/pi/cond/Equal/Switch:1target/pi/cond/Equal/Switch:13
target/pi/sub:0 target/pi/cond/cond/sub/Switch:0
Ф
target/pi/cond/cond_text_1target/pi/cond/pred_id:0target/pi/cond/switch_f:0*І

target/pi/cond/Equal_1/y:0
target/pi/cond/Equal_1:0
target/pi/cond/Exp_1/Switch:0
target/pi/cond/Exp_1:0
target/pi/cond/Maximum_1/y:0
target/pi/cond/Maximum_1:0
target/pi/cond/Minimum:0
target/pi/cond/Pow/x:0
target/pi/cond/Pow:0
(target/pi/cond/Sum_1/reduction_indices:0
target/pi/cond/Sum_1:0
"target/pi/cond/cond_1/Log/Switch:1
target/pi/cond/cond_1/Log:0
target/pi/cond/cond_1/Merge:0
target/pi/cond/cond_1/Merge:1
"target/pi/cond/cond_1/Pow/Switch:0
target/pi/cond/cond_1/Pow:0
target/pi/cond/cond_1/Switch:0
target/pi/cond/cond_1/Switch:1
target/pi/cond/cond_1/pred_id:0
"target/pi/cond/cond_1/sub/Switch:0
target/pi/cond/cond_1/sub/x:0
target/pi/cond/cond_1/sub:0
target/pi/cond/cond_1/sub_1/y:0
target/pi/cond/cond_1/sub_1:0
target/pi/cond/cond_1/sub_2/x:0
target/pi/cond/cond_1/sub_2:0
 target/pi/cond/cond_1/switch_f:0
 target/pi/cond/cond_1/switch_t:0
target/pi/cond/cond_1/truediv:0
target/pi/cond/pred_id:0
target/pi/cond/sub/Switch:0
target/pi/cond/sub/x:0
target/pi/cond/sub:0
target/pi/cond/switch_f:0
target/pi/cond/truediv/x:0
target/pi/cond/truediv:0
target/pi/sub:0
target/pi/sub_5:02
target/pi/sub_5:0target/pi/cond/Exp_1/Switch:04
target/pi/cond/pred_id:0target/pi/cond/pred_id:0.
target/pi/sub:0target/pi/cond/sub/Switch:02њ
Ј
target/pi/cond/cond_1/cond_texttarget/pi/cond/cond_1/pred_id:0 target/pi/cond/cond_1/switch_t:0 *д
target/pi/cond/Maximum_1:0
"target/pi/cond/cond_1/Log/Switch:1
target/pi/cond/cond_1/Log:0
target/pi/cond/cond_1/pred_id:0
 target/pi/cond/cond_1/switch_t:0@
target/pi/cond/Maximum_1:0"target/pi/cond/cond_1/Log/Switch:1B
target/pi/cond/cond_1/pred_id:0target/pi/cond/cond_1/pred_id:02┤
▒
!target/pi/cond/cond_1/cond_text_1target/pi/cond/cond_1/pred_id:0 target/pi/cond/cond_1/switch_f:0*╚
target/pi/cond/Maximum_1:0
"target/pi/cond/cond_1/Pow/Switch:0
target/pi/cond/cond_1/Pow:0
target/pi/cond/cond_1/pred_id:0
"target/pi/cond/cond_1/sub/Switch:0
target/pi/cond/cond_1/sub/x:0
target/pi/cond/cond_1/sub:0
target/pi/cond/cond_1/sub_1/y:0
target/pi/cond/cond_1/sub_1:0
target/pi/cond/cond_1/sub_2/x:0
target/pi/cond/cond_1/sub_2:0
 target/pi/cond/cond_1/switch_f:0
target/pi/cond/cond_1/truediv:0
target/pi/cond/sub/Switch:0
target/pi/sub:05
target/pi/sub:0"target/pi/cond/cond_1/sub/Switch:0@
target/pi/cond/Maximum_1:0"target/pi/cond/cond_1/Pow/Switch:0B
target/pi/cond/cond_1/pred_id:0target/pi/cond/cond_1/pred_id:0:
target/pi/cond/sub/Switch:0target/pi/cond/sub/Switch:0"
train_op

Adam
Adam_1"┬x
	variables┤x▒x
Є
main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:021main/pi/dense/kernel/Initializer/random_uniform:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08
Ј
main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:023main/pi/dense_1/kernel/Initializer/random_uniform:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08
Ј
main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08
Ј
main/pi/dense_3/kernel:0main/pi/dense_3/kernel/Assignmain/pi/dense_3/kernel/read:023main/pi/dense_3/kernel/Initializer/random_uniform:08
~
main/pi/dense_3/bias:0main/pi/dense_3/bias/Assignmain/pi/dense_3/bias/read:02(main/pi/dense_3/bias/Initializer/zeros:08
Є
main/q1/dense/kernel:0main/q1/dense/kernel/Assignmain/q1/dense/kernel/read:021main/q1/dense/kernel/Initializer/random_uniform:08
v
main/q1/dense/bias:0main/q1/dense/bias/Assignmain/q1/dense/bias/read:02&main/q1/dense/bias/Initializer/zeros:08
Ј
main/q1/dense_1/kernel:0main/q1/dense_1/kernel/Assignmain/q1/dense_1/kernel/read:023main/q1/dense_1/kernel/Initializer/random_uniform:08
~
main/q1/dense_1/bias:0main/q1/dense_1/bias/Assignmain/q1/dense_1/bias/read:02(main/q1/dense_1/bias/Initializer/zeros:08
Ј
main/q1/dense_2/kernel:0main/q1/dense_2/kernel/Assignmain/q1/dense_2/kernel/read:023main/q1/dense_2/kernel/Initializer/random_uniform:08
~
main/q1/dense_2/bias:0main/q1/dense_2/bias/Assignmain/q1/dense_2/bias/read:02(main/q1/dense_2/bias/Initializer/zeros:08
Є
main/q2/dense/kernel:0main/q2/dense/kernel/Assignmain/q2/dense/kernel/read:021main/q2/dense/kernel/Initializer/random_uniform:08
v
main/q2/dense/bias:0main/q2/dense/bias/Assignmain/q2/dense/bias/read:02&main/q2/dense/bias/Initializer/zeros:08
Ј
main/q2/dense_1/kernel:0main/q2/dense_1/kernel/Assignmain/q2/dense_1/kernel/read:023main/q2/dense_1/kernel/Initializer/random_uniform:08
~
main/q2/dense_1/bias:0main/q2/dense_1/bias/Assignmain/q2/dense_1/bias/read:02(main/q2/dense_1/bias/Initializer/zeros:08
Ј
main/q2/dense_2/kernel:0main/q2/dense_2/kernel/Assignmain/q2/dense_2/kernel/read:023main/q2/dense_2/kernel/Initializer/random_uniform:08
~
main/q2/dense_2/bias:0main/q2/dense_2/bias/Assignmain/q2/dense_2/bias/read:02(main/q2/dense_2/bias/Initializer/zeros:08
Ѓ
main/v/dense/kernel:0main/v/dense/kernel/Assignmain/v/dense/kernel/read:020main/v/dense/kernel/Initializer/random_uniform:08
r
main/v/dense/bias:0main/v/dense/bias/Assignmain/v/dense/bias/read:02%main/v/dense/bias/Initializer/zeros:08
І
main/v/dense_1/kernel:0main/v/dense_1/kernel/Assignmain/v/dense_1/kernel/read:022main/v/dense_1/kernel/Initializer/random_uniform:08
z
main/v/dense_1/bias:0main/v/dense_1/bias/Assignmain/v/dense_1/bias/read:02'main/v/dense_1/bias/Initializer/zeros:08
І
main/v/dense_2/kernel:0main/v/dense_2/kernel/Assignmain/v/dense_2/kernel/read:022main/v/dense_2/kernel/Initializer/random_uniform:08
z
main/v/dense_2/bias:0main/v/dense_2/bias/Assignmain/v/dense_2/bias/read:02'main/v/dense_2/bias/Initializer/zeros:08
Ј
target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:023target/pi/dense/kernel/Initializer/random_uniform:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08
Ќ
target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:025target/pi/dense_1/kernel/Initializer/random_uniform:08
є
target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08
Ќ
target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08
є
target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08
Ќ
target/pi/dense_3/kernel:0target/pi/dense_3/kernel/Assigntarget/pi/dense_3/kernel/read:025target/pi/dense_3/kernel/Initializer/random_uniform:08
є
target/pi/dense_3/bias:0target/pi/dense_3/bias/Assigntarget/pi/dense_3/bias/read:02*target/pi/dense_3/bias/Initializer/zeros:08
Ј
target/q1/dense/kernel:0target/q1/dense/kernel/Assigntarget/q1/dense/kernel/read:023target/q1/dense/kernel/Initializer/random_uniform:08
~
target/q1/dense/bias:0target/q1/dense/bias/Assigntarget/q1/dense/bias/read:02(target/q1/dense/bias/Initializer/zeros:08
Ќ
target/q1/dense_1/kernel:0target/q1/dense_1/kernel/Assigntarget/q1/dense_1/kernel/read:025target/q1/dense_1/kernel/Initializer/random_uniform:08
є
target/q1/dense_1/bias:0target/q1/dense_1/bias/Assigntarget/q1/dense_1/bias/read:02*target/q1/dense_1/bias/Initializer/zeros:08
Ќ
target/q1/dense_2/kernel:0target/q1/dense_2/kernel/Assigntarget/q1/dense_2/kernel/read:025target/q1/dense_2/kernel/Initializer/random_uniform:08
є
target/q1/dense_2/bias:0target/q1/dense_2/bias/Assigntarget/q1/dense_2/bias/read:02*target/q1/dense_2/bias/Initializer/zeros:08
Ј
target/q2/dense/kernel:0target/q2/dense/kernel/Assigntarget/q2/dense/kernel/read:023target/q2/dense/kernel/Initializer/random_uniform:08
~
target/q2/dense/bias:0target/q2/dense/bias/Assigntarget/q2/dense/bias/read:02(target/q2/dense/bias/Initializer/zeros:08
Ќ
target/q2/dense_1/kernel:0target/q2/dense_1/kernel/Assigntarget/q2/dense_1/kernel/read:025target/q2/dense_1/kernel/Initializer/random_uniform:08
є
target/q2/dense_1/bias:0target/q2/dense_1/bias/Assigntarget/q2/dense_1/bias/read:02*target/q2/dense_1/bias/Initializer/zeros:08
Ќ
target/q2/dense_2/kernel:0target/q2/dense_2/kernel/Assigntarget/q2/dense_2/kernel/read:025target/q2/dense_2/kernel/Initializer/random_uniform:08
є
target/q2/dense_2/bias:0target/q2/dense_2/bias/Assigntarget/q2/dense_2/bias/read:02*target/q2/dense_2/bias/Initializer/zeros:08
І
target/v/dense/kernel:0target/v/dense/kernel/Assigntarget/v/dense/kernel/read:022target/v/dense/kernel/Initializer/random_uniform:08
z
target/v/dense/bias:0target/v/dense/bias/Assigntarget/v/dense/bias/read:02'target/v/dense/bias/Initializer/zeros:08
Њ
target/v/dense_1/kernel:0target/v/dense_1/kernel/Assigntarget/v/dense_1/kernel/read:024target/v/dense_1/kernel/Initializer/random_uniform:08
ѓ
target/v/dense_1/bias:0target/v/dense_1/bias/Assigntarget/v/dense_1/bias/read:02)target/v/dense_1/bias/Initializer/zeros:08
Њ
target/v/dense_2/kernel:0target/v/dense_2/kernel/Assigntarget/v/dense_2/kernel/read:024target/v/dense_2/kernel/Initializer/random_uniform:08
ѓ
target/v/dense_2/bias:0target/v/dense_2/bias/Assigntarget/v/dense_2/bias/read:02)target/v/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
љ
main/pi/dense/kernel/Adam:0 main/pi/dense/kernel/Adam/Assign main/pi/dense/kernel/Adam/read:02-main/pi/dense/kernel/Adam/Initializer/zeros:0
ў
main/pi/dense/kernel/Adam_1:0"main/pi/dense/kernel/Adam_1/Assign"main/pi/dense/kernel/Adam_1/read:02/main/pi/dense/kernel/Adam_1/Initializer/zeros:0
ѕ
main/pi/dense/bias/Adam:0main/pi/dense/bias/Adam/Assignmain/pi/dense/bias/Adam/read:02+main/pi/dense/bias/Adam/Initializer/zeros:0
љ
main/pi/dense/bias/Adam_1:0 main/pi/dense/bias/Adam_1/Assign main/pi/dense/bias/Adam_1/read:02-main/pi/dense/bias/Adam_1/Initializer/zeros:0
ў
main/pi/dense_1/kernel/Adam:0"main/pi/dense_1/kernel/Adam/Assign"main/pi/dense_1/kernel/Adam/read:02/main/pi/dense_1/kernel/Adam/Initializer/zeros:0
а
main/pi/dense_1/kernel/Adam_1:0$main/pi/dense_1/kernel/Adam_1/Assign$main/pi/dense_1/kernel/Adam_1/read:021main/pi/dense_1/kernel/Adam_1/Initializer/zeros:0
љ
main/pi/dense_1/bias/Adam:0 main/pi/dense_1/bias/Adam/Assign main/pi/dense_1/bias/Adam/read:02-main/pi/dense_1/bias/Adam/Initializer/zeros:0
ў
main/pi/dense_1/bias/Adam_1:0"main/pi/dense_1/bias/Adam_1/Assign"main/pi/dense_1/bias/Adam_1/read:02/main/pi/dense_1/bias/Adam_1/Initializer/zeros:0
ў
main/pi/dense_2/kernel/Adam:0"main/pi/dense_2/kernel/Adam/Assign"main/pi/dense_2/kernel/Adam/read:02/main/pi/dense_2/kernel/Adam/Initializer/zeros:0
а
main/pi/dense_2/kernel/Adam_1:0$main/pi/dense_2/kernel/Adam_1/Assign$main/pi/dense_2/kernel/Adam_1/read:021main/pi/dense_2/kernel/Adam_1/Initializer/zeros:0
љ
main/pi/dense_2/bias/Adam:0 main/pi/dense_2/bias/Adam/Assign main/pi/dense_2/bias/Adam/read:02-main/pi/dense_2/bias/Adam/Initializer/zeros:0
ў
main/pi/dense_2/bias/Adam_1:0"main/pi/dense_2/bias/Adam_1/Assign"main/pi/dense_2/bias/Adam_1/read:02/main/pi/dense_2/bias/Adam_1/Initializer/zeros:0
ў
main/pi/dense_3/kernel/Adam:0"main/pi/dense_3/kernel/Adam/Assign"main/pi/dense_3/kernel/Adam/read:02/main/pi/dense_3/kernel/Adam/Initializer/zeros:0
а
main/pi/dense_3/kernel/Adam_1:0$main/pi/dense_3/kernel/Adam_1/Assign$main/pi/dense_3/kernel/Adam_1/read:021main/pi/dense_3/kernel/Adam_1/Initializer/zeros:0
љ
main/pi/dense_3/bias/Adam:0 main/pi/dense_3/bias/Adam/Assign main/pi/dense_3/bias/Adam/read:02-main/pi/dense_3/bias/Adam/Initializer/zeros:0
ў
main/pi/dense_3/bias/Adam_1:0"main/pi/dense_3/bias/Adam_1/Assign"main/pi/dense_3/bias/Adam_1/read:02/main/pi/dense_3/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
љ
main/q1/dense/kernel/Adam:0 main/q1/dense/kernel/Adam/Assign main/q1/dense/kernel/Adam/read:02-main/q1/dense/kernel/Adam/Initializer/zeros:0
ў
main/q1/dense/kernel/Adam_1:0"main/q1/dense/kernel/Adam_1/Assign"main/q1/dense/kernel/Adam_1/read:02/main/q1/dense/kernel/Adam_1/Initializer/zeros:0
ѕ
main/q1/dense/bias/Adam:0main/q1/dense/bias/Adam/Assignmain/q1/dense/bias/Adam/read:02+main/q1/dense/bias/Adam/Initializer/zeros:0
љ
main/q1/dense/bias/Adam_1:0 main/q1/dense/bias/Adam_1/Assign main/q1/dense/bias/Adam_1/read:02-main/q1/dense/bias/Adam_1/Initializer/zeros:0
ў
main/q1/dense_1/kernel/Adam:0"main/q1/dense_1/kernel/Adam/Assign"main/q1/dense_1/kernel/Adam/read:02/main/q1/dense_1/kernel/Adam/Initializer/zeros:0
а
main/q1/dense_1/kernel/Adam_1:0$main/q1/dense_1/kernel/Adam_1/Assign$main/q1/dense_1/kernel/Adam_1/read:021main/q1/dense_1/kernel/Adam_1/Initializer/zeros:0
љ
main/q1/dense_1/bias/Adam:0 main/q1/dense_1/bias/Adam/Assign main/q1/dense_1/bias/Adam/read:02-main/q1/dense_1/bias/Adam/Initializer/zeros:0
ў
main/q1/dense_1/bias/Adam_1:0"main/q1/dense_1/bias/Adam_1/Assign"main/q1/dense_1/bias/Adam_1/read:02/main/q1/dense_1/bias/Adam_1/Initializer/zeros:0
ў
main/q1/dense_2/kernel/Adam:0"main/q1/dense_2/kernel/Adam/Assign"main/q1/dense_2/kernel/Adam/read:02/main/q1/dense_2/kernel/Adam/Initializer/zeros:0
а
main/q1/dense_2/kernel/Adam_1:0$main/q1/dense_2/kernel/Adam_1/Assign$main/q1/dense_2/kernel/Adam_1/read:021main/q1/dense_2/kernel/Adam_1/Initializer/zeros:0
љ
main/q1/dense_2/bias/Adam:0 main/q1/dense_2/bias/Adam/Assign main/q1/dense_2/bias/Adam/read:02-main/q1/dense_2/bias/Adam/Initializer/zeros:0
ў
main/q1/dense_2/bias/Adam_1:0"main/q1/dense_2/bias/Adam_1/Assign"main/q1/dense_2/bias/Adam_1/read:02/main/q1/dense_2/bias/Adam_1/Initializer/zeros:0
љ
main/q2/dense/kernel/Adam:0 main/q2/dense/kernel/Adam/Assign main/q2/dense/kernel/Adam/read:02-main/q2/dense/kernel/Adam/Initializer/zeros:0
ў
main/q2/dense/kernel/Adam_1:0"main/q2/dense/kernel/Adam_1/Assign"main/q2/dense/kernel/Adam_1/read:02/main/q2/dense/kernel/Adam_1/Initializer/zeros:0
ѕ
main/q2/dense/bias/Adam:0main/q2/dense/bias/Adam/Assignmain/q2/dense/bias/Adam/read:02+main/q2/dense/bias/Adam/Initializer/zeros:0
љ
main/q2/dense/bias/Adam_1:0 main/q2/dense/bias/Adam_1/Assign main/q2/dense/bias/Adam_1/read:02-main/q2/dense/bias/Adam_1/Initializer/zeros:0
ў
main/q2/dense_1/kernel/Adam:0"main/q2/dense_1/kernel/Adam/Assign"main/q2/dense_1/kernel/Adam/read:02/main/q2/dense_1/kernel/Adam/Initializer/zeros:0
а
main/q2/dense_1/kernel/Adam_1:0$main/q2/dense_1/kernel/Adam_1/Assign$main/q2/dense_1/kernel/Adam_1/read:021main/q2/dense_1/kernel/Adam_1/Initializer/zeros:0
љ
main/q2/dense_1/bias/Adam:0 main/q2/dense_1/bias/Adam/Assign main/q2/dense_1/bias/Adam/read:02-main/q2/dense_1/bias/Adam/Initializer/zeros:0
ў
main/q2/dense_1/bias/Adam_1:0"main/q2/dense_1/bias/Adam_1/Assign"main/q2/dense_1/bias/Adam_1/read:02/main/q2/dense_1/bias/Adam_1/Initializer/zeros:0
ў
main/q2/dense_2/kernel/Adam:0"main/q2/dense_2/kernel/Adam/Assign"main/q2/dense_2/kernel/Adam/read:02/main/q2/dense_2/kernel/Adam/Initializer/zeros:0
а
main/q2/dense_2/kernel/Adam_1:0$main/q2/dense_2/kernel/Adam_1/Assign$main/q2/dense_2/kernel/Adam_1/read:021main/q2/dense_2/kernel/Adam_1/Initializer/zeros:0
љ
main/q2/dense_2/bias/Adam:0 main/q2/dense_2/bias/Adam/Assign main/q2/dense_2/bias/Adam/read:02-main/q2/dense_2/bias/Adam/Initializer/zeros:0
ў
main/q2/dense_2/bias/Adam_1:0"main/q2/dense_2/bias/Adam_1/Assign"main/q2/dense_2/bias/Adam_1/read:02/main/q2/dense_2/bias/Adam_1/Initializer/zeros:0
ї
main/v/dense/kernel/Adam:0main/v/dense/kernel/Adam/Assignmain/v/dense/kernel/Adam/read:02,main/v/dense/kernel/Adam/Initializer/zeros:0
ћ
main/v/dense/kernel/Adam_1:0!main/v/dense/kernel/Adam_1/Assign!main/v/dense/kernel/Adam_1/read:02.main/v/dense/kernel/Adam_1/Initializer/zeros:0
ё
main/v/dense/bias/Adam:0main/v/dense/bias/Adam/Assignmain/v/dense/bias/Adam/read:02*main/v/dense/bias/Adam/Initializer/zeros:0
ї
main/v/dense/bias/Adam_1:0main/v/dense/bias/Adam_1/Assignmain/v/dense/bias/Adam_1/read:02,main/v/dense/bias/Adam_1/Initializer/zeros:0
ћ
main/v/dense_1/kernel/Adam:0!main/v/dense_1/kernel/Adam/Assign!main/v/dense_1/kernel/Adam/read:02.main/v/dense_1/kernel/Adam/Initializer/zeros:0
ю
main/v/dense_1/kernel/Adam_1:0#main/v/dense_1/kernel/Adam_1/Assign#main/v/dense_1/kernel/Adam_1/read:020main/v/dense_1/kernel/Adam_1/Initializer/zeros:0
ї
main/v/dense_1/bias/Adam:0main/v/dense_1/bias/Adam/Assignmain/v/dense_1/bias/Adam/read:02,main/v/dense_1/bias/Adam/Initializer/zeros:0
ћ
main/v/dense_1/bias/Adam_1:0!main/v/dense_1/bias/Adam_1/Assign!main/v/dense_1/bias/Adam_1/read:02.main/v/dense_1/bias/Adam_1/Initializer/zeros:0
ћ
main/v/dense_2/kernel/Adam:0!main/v/dense_2/kernel/Adam/Assign!main/v/dense_2/kernel/Adam/read:02.main/v/dense_2/kernel/Adam/Initializer/zeros:0
ю
main/v/dense_2/kernel/Adam_1:0#main/v/dense_2/kernel/Adam_1/Assign#main/v/dense_2/kernel/Adam_1/read:020main/v/dense_2/kernel/Adam_1/Initializer/zeros:0
ї
main/v/dense_2/bias/Adam:0main/v/dense_2/bias/Adam/Assignmain/v/dense_2/bias/Adam/read:02,main/v/dense_2/bias/Adam/Initializer/zeros:0
ћ
main/v/dense_2/bias/Adam_1:0!main/v/dense_2/bias/Adam_1/Assign!main/v/dense_2/bias/Adam_1/read:02.main/v/dense_2/bias/Adam_1/Initializer/zeros:0*џ
serving_defaultє

alpha
Placeholder_5:0 
)
x$
Placeholder:0         I
+
a&
Placeholder_1:0         

q
Placeholder_6:0 (
v#
main/v/Squeeze:0         *
q1$
main/q1/Squeeze:0         '
mu!

main/mul:0         *
q2$
main/q2/Squeeze:0         )
pi#
main/mul_1:0         tensorflow/serving/predict