#include "header/addr.h"

void Addr::Set_addr(int port)
{
    addrlen = sizeof(addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
}
