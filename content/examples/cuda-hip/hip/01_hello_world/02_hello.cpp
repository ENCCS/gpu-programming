//
// nvcc 02_hello.cu # on nvidia platorms
// cc -xhip 02_hello.cu # on amd platforms
//
#include <stdio.h>

int main(int argc, const char * argv[])
{
	printf("\n----------------------\n");
    printf("Hello World from CPU!\n");
	printf("----------------------\n\n");

    return 0;
}
