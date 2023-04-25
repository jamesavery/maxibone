#include<iostream>
#include<stdint.h>

int main() {
    /*
    This class is used to print out the code of the type.
    This is handy when debugging the templated type at runtime.
    */

    std::cout << "int8 " << typeid(int8_t).name() << std::endl;
    std::cout << "int16 " << typeid(int16_t).name() << std::endl;
    std::cout << "int32 " << typeid(int32_t).name() << std::endl;
    std::cout << "int64 " << typeid(int64_t).name() << std::endl;
    std::cout << "int128 " << typeid(__int128_t).name() << std::endl;
    
    std::cout << "uint8 " << typeid(uint8_t).name() << std::endl;
    std::cout << "uint16 " << typeid(uint16_t).name() << std::endl;
    std::cout << "uint32 " << typeid(uint32_t).name() << std::endl;
    std::cout << "uint64 " << typeid(uint64_t).name() << std::endl;
    std::cout << "uint128 " << typeid(__uint128_t).name() << std::endl;
    
    std::cout << "float " << typeid(float).name() << std::endl;
    std::cout << "double " << typeid(double).name() << std::endl;
}