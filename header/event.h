#ifndef __EVENT_H_
#define __EVENT_H_

#include <string>
#include <cstring>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include "sockset.h"
#include "size.h"



class Event : public Sock_set
{
    
    public:
        Event() {}
        ~Event() {}
        void Create_event();
        void Set_event();
        void E_wait();

    protected:
        struct epoll_event ev, *events;
        int epoll_fd;
        int user_fds[SIZE_CONST::BUF_SIZE];
        int event_cnt;
    
};

#endif