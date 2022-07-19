#include "header/server.h"

void Server::Main_Func()
{
    Server::Set_listen_fd();
    Server::Set_addr(port);
    Server::Bind_addr();
    Server::Listen();
    Server::Create_event();
    Server::Set_event();

    while(true)
    {
        Server::E_wait();
        for(i=0; i<event_cnt; i++)
        {
            if(events[i].data.fd == listen_fd)
            {
                Accept();
            }
            else
            {
                user_data = (udata*)events[i].data.ptr;
                memset(buf, 0x00, SIZE_CONST::BUF_SIZE);
                read_cnt = read(user_data->fd, buf, SIZE_CONST::BUF_SIZE);
                if(read_cnt<=0)
                {
                    Disconnect();
                }
                else    
                {
                    cout<<"buf:"<<buf<<endl;
                    string clnt_msg(buf);
                    Server::Recv_main(user_data, clnt_msg);
                }
                
            }
        }
    }
}



void Server::Accept()
{
    clnt_len = sizeof(struct sockaddr);
    client_fd = accept(listen_fd, (struct sockaddr *)&client_addr, &clnt_len);
    user_fds[client_fd] = 1;
    
    user_data = new udata;
    user_data->fd = client_fd;

    ev.events = EPOLLIN;
    ev.data.ptr = user_data;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);

}

void Server::Disconnect()
{
    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, user_data->fd, events);
    close(user_data->fd);
    user_fds[user_data->fd] = -1;
    delete user_data;
}