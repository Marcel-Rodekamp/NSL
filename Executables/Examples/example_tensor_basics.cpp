// include iostream for printing to terminal
#include <iostream>
// include NSL
#include "NSL.hpp"

int main(){
    /* Creation of a Tensor
     * =================================
     * Suppose you want to create a Tensor of two dimensions (matrix) with 
     * 16 elements each being of type double. This can be done using the 
     * D-dimensional constructor:
     * ``` NSL::Tensor<datatype> TensorName(dim0Size,dim1Size,...);```
     * Each element is automatically filled with zeros.
     */
    NSL::Tensor<double> mat(4,4);

    /*
     *
     * The tensor can now be printed to the terminal using `std::cout`.
     * You should find an output similar to:
     * ```
     * 0 0 0 0
     * 0 0 0 0
     * 0 0 0 0
     * 0 0 0 0
     *[ CPUDoubleType{4} ]
     * ```
     * */
    std::cout << "The matrix `mat` after construction is: " << std::endl
              << mat 
              << std::endl;

    std::cout << std::endl << "=================================" << std::endl << std::endl;

    /* Factory of Tensors
     * =================================
     * There are many different methods to create tensors. 
     * 1. Factory Functions
     *     These are functions that construct Tensors with specific properties
     * 2. In place
     *     These are Tensor methods which change the data of a previously
     *     Constructed Tensor.
     */

    /* @1. Factory functions
     * \todo There is only zeros_like implemented sofar...
     */
    
    /* @2. In place 
     * Uniformly draw random variables between 0 and 1 to generate some non
     * zero elements in our matrix
     */
    mat.rand();
    
    /*
     *
     * You should find an output like this:
     * ```
     *  0.3441  0.3064  0.1697  0.9121
     *  0.7874  0.6885  0.5695  0.6014
     *  0.3156  0.1017  0.5897  0.8287
     *  0.3183  0.1287  0.3599  0.9350
     * [ CPUDoubleType{4,4} ]
     * ```
     * */
    std::cout << "The matrix `mat` after construction is: " << std::endl
              << mat 
              << std::endl;

    std::cout << std::endl << "=================================" << std::endl << std::endl;
    
    /* Accessing the Tensor Elements
     * =================================
     * The Tensor provides multiple ways of accessing it's elements, i.e.
     * the underlying data.
     *
     * 1. Random Element Access
     * 1.1. Per dimension
     *      Access each element by a combination of indices in the tensor
     * 1.2. Element loop
     *      Access each element by a index corresponding to the linearized
     *      momory layout
     * 1.3. Slice Access
     *      Access whole slices through a Tensor using indices & `NSL::Slice`
     *      objects.
     * 2. Pointer Access
     *      Access the underlying C-pointer (momory address) directly
     * 3. Statistics
     *      Get the dimension,shape etc.
     */
    
    /* @1.1. Random Access Per dimension
     * Given the two indices (i,j) one can address a matrix element.
     * This is done using the operator `NSL::Tensor::operator()`.
     * It can be used for reading and writing
     * */
    std::cout << "The (0,1) element of the matrix `mat` is: "
              << mat(0,1) 
              << std::endl;
    mat(0,1) = 3.141592;
    std::cout << "The (0,1) element of the matrix `mat` changed to: "
              << mat(0,1) 
              << std::endl;

    std::cout << std::endl << "=================================" << std::endl << std::endl;


    /* @1.2. Element loop
     * Given the two indices (i,j) one can translate it to a linearized 
     * index \f$ l = i + j \cdot N_0 \f$. This is the index used to identify
     * a element in the computer memory. Sometimes one wants to loop over
     * all elements of a Tensor explicitly and the dimension doesn't matter.
     * Then the operator `NSL::Tensor::operator[]` can be used like this:
     */
    std::cout << "The 3rd element of the matrix `mat` in memory is: "
              << mat[2]
              << std::endl;
    mat[2] = 3.141592;
    std::cout << "The 3rd element of the matrix `mat` changed to: "
              << mat[2]
              << std::endl;

    std::cout << std::endl << "=================================" << std::endl << std::endl;

    /* @1.3. Slice Access
     * Sometimes one wants to change a entire row/column of the matrix at 
     * hand. Then the Slice access comes in handy using the operator 
     * `NSL::Tensor::operator()`
     */ 
    std::cout << "The 1st row of the matrix `mat` is: " << std::endl
              << mat(0,NSL::Slice())
              << std::endl;

    std::cout << std::endl << "=================================" << std::endl << std::endl;

    /* 
     * Further we can only change a portion of a given slice, let's set the 
     * center elements to zero such that our matrix reads:
     * ```
     *  *  *  *  * 
     *  *  0  0  *
     *  *  0  0  *
     *  *  *  *  *
     * [ CPUDoubleType{4,4} ]
     * ```
     * */
    mat(NSL::Slice(/*start*/1,/*stop*/3),NSL::Slice(/*start*/1,/*stop*/3)) = 0.;
    std::cout << "The center elements of the matrix `mat` is set to zero: " << std::endl
              << mat << std::endl;

    std::cout << std::endl << "=================================" << std::endl << std::endl;

    /* @2. Pointer Access
     * If the underlying address space needs to be accessed one can utilize
     * the standard C++ method `NSL::Tensor::data()`
     * */
    std::cout << "The address of the matrix `mat` is: "
              << mat.data() 
              << std::endl;

    std::cout << std::endl << "=================================" << std::endl << std::endl;

    /* 3. Stats
     * Last but not least we can query some defining propperties of the
     * Tensor:
     * */
    std::cout << "The matrix `mat` has: " << std::endl
              << "\t * dimension: " << mat.dim() << std::endl
              << "\t * elements per row: " << mat.shape(0) << std::endl
              << "\t * elements per col: " << mat.shape(1) << std::endl
              << "\t * total number of elements: " << mat.numel()
              << std::endl;

    std::cout << std::endl << "=================================" << std::endl << std::endl;

    /* Tensor operations
     * =================================
     * A lot of linear algebra and manipulation methods are available.
     * Typically all Tensor member functions are considered in place 
     * with pnly a few exceptions. 
     * The same operations also exist in the `NSL::LinAlg` name space 
     * which should be used for not-inplace computation (careful, these 
     * perform an explicit copy of the data)
     */
    // 1. Tensor arithmetic like (elementwise) addition
    NSL::Tensor<double> mat2(4,4); mat2.rand();
    auto matRes = mat + mat2; // creates a new tensor with new memory
    mat += mat2; // uses the same memory space of mat

    std::cout << "Adding another random matrix to `mat` results in: " << std::endl
              << mat 
              << std::endl;
                 
    std::cout << std::endl << "=================================" << std::endl << std::endl;

    // 2. Tensor arithmetic like (elementwise) multiplication
    NSL::Tensor<double> mat3(4,4); mat3.rand();
    auto matProd = mat(0, NSL::Slice()) * mat3; // creates a new tensor with new memory

    std::cout << "Multiplying another random matrix to `mat` results in: " << std::endl
              << mat(0, NSL::Slice())
              << mat3
              << matProd
              << std::endl;
                 
    std::cout << std::endl << "=================================" << std::endl << std::endl;

    // 3. Elementwise operations like exponentiation
    auto matExp = NSL::LinAlg::exp(mat); // creates a new tensor with new memory
    mat.exp(); // uses the same memory space as mat
    
    std::cout << "Exponentiating the matrix `mat` elementwise results in: " << std::endl
              << mat 
              << std::endl;
               
    std::cout << std::endl << "=================================" << std::endl << std::endl;

    /* Memory location
     * =================================
     * By default, the Tensor copies are shallow that means a new Tensor
     * is created but with the same memory as it's source.
     * Only specific methods are ment to perform a deepcopy that
     * means a new tensor with new memory is created and the actualt data
     * values are copied.
     * The following shows a non extinsive list of shallow/deep copies
     * */

    // Shallow copy: Copy constructor
    NSL::Tensor<double> mat_cc(mat);
    std::cout << std::boolalpha << "`mat_cc` and `mat` share memory: " << (mat_cc.data() == mat.data()) << std::endl;

    std::cout << "=================================" << std::endl;

    // Shallow copy: Assignment (copy) constructor
    NSL::Tensor<double> mat_acc = mat;
    std::cout << std::boolalpha << "`mat_acc` and `mat` share memory: " << (mat_acc.data() == mat.data()) << std::endl;

    std::cout << "=================================" << std::endl;

    // Shallow copy: translation to Torch
    // by copy constructor
    torch::Tensor torch_mat_cc(mat);
    std::cout << std::boolalpha << "`torch_mat_cc` and `mat` share memory: " << (torch_mat_cc.data_ptr<double>() == mat.data()) << std::endl;

    std::cout << "=================================" << std::endl;

    // by assignment 
    torch::Tensor torch_mat_acc = mat;
    std::cout << std::boolalpha << "`torch_mat_acc` and `mat` share memory: " << (torch_mat_acc.data_ptr<double>() == mat.data()) << std::endl;

    std::cout << "=================================" << std::endl;

    // Deep copy: deep copy constructor
    NSL::Tensor<double> mat_dcc(mat,true); // if false is passed a shallow copy is performed
    std::cout << std::boolalpha << "`mat_dcc` and `mat` don't share memory: " << (mat_dcc.data() == mat.data()) << std::endl;

    std::cout << "=================================" << std::endl;

    // Deep copy: Assignment
    NSL::Tensor<double> mat_asn(4,4);
    mat_asn = mat;
    std::cout << std::boolalpha << "`mat_asn` and `mat` don't share memory: " << (mat_asn.data() == mat.data()) << std::endl;

    std::cout << "=================================" << std::endl;
    
    // Technically the following things perform a deep copy if you like
    // Move construction, invalidates/destructs the source
    const double * address_bak = mat.data();
    NSL::Tensor<double> mat_mv(std::move(mat));
    std::cout << std::boolalpha << "`mat_mv` has the memory of `mat`: " << (mat_mv.data() == address_bak) << std::endl;
    // any call on mat is not valid past this.



    return EXIT_SUCCESS;
}
