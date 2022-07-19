#include <iostream>
#include "header/server.h"
using namespace std;

int main(int argc, char * argv[])
{

    if (argc != 2)
    {
        cout<<" Usage: "<<argv[0]<<" <port>\n";
        exit(1);
    }
    int port = atoi(argv[1]);
    Server serv(port);

    serv.Main_Func();
    

    return 0;
}