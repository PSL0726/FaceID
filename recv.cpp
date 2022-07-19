#include "header/recv.h"
#include "header/database.h"
#include "header/udata.h"
#include "header/size.h"

using namespace std;

void Recv::login()
{
    cout<<clnt_msg<<endl;
    vector<string> check_msg = Recv::split(clnt_msg, '/');
    sprintf(query, "SELECT id, pw FROM user where id='%s'", check_msg[1].c_str());
    if(mysql_query(&conn, query)!=0)
    {
        write(user_data->fd, "NO", sizeof("NO"));
        puts("login_error");
    }
    else
    {
        sql_result = mysql_store_result(&conn);
        sql_row = mysql_fetch_row(sql_result);
        cout<<"password: "<<sql_row[0]<<endl;
        if(strcmp(check_msg[2].c_str(), sql_row[1])==0)
        {
            write(user_data->fd, "OK", sizeof("OK"));
            puts("login_sucess"); 
            cout<<sql_row[0]<<" "<<sql_row[1]<<endl;      
            strcpy(user_data->name, sql_row[1]);
            cout<<user_data->name<<endl;
        }
        else
        {
            write(user_data->fd, "NO", sizeof("NO"));
            puts("login_error");
        }        
        mysql_free_result(sql_result);
    }
}

void Recv::id_check()
{
    cout<<clnt_msg<<endl;
    cout<<"아이디체크안임 ㅇㅇ\n";
    vector<string> check_msg = Recv::split(clnt_msg, '/');
    for (int i = 0; i < check_msg.size(); i++)
    {
        cout << check_msg[i] << endl;
    }
    sprintf(query, "SELECT * FROM user WHERE id = '%s'", check_msg[1].c_str());
    if(mysql_query(&conn,query)!=0)
    {
        write(user_data->fd,"OK",sizeof("OK"));
        cout<<"OK"<<endl;
    } 
    else
    {
        write(user_data->fd,"NO",sizeof("NO"));
        cout<<"NO"<<endl;
    }
}

void Recv::sign_up()
{
    sql_result = mysql_store_result(&conn);  
    cout<<clnt_msg<<endl;
    vector<string> sign_msg = Recv::split(clnt_msg, '/');    
   
    sprintf(query, "INSERT INTO user VALUES('%s','%s','%s','%s')", sign_msg[1].c_str(), sign_msg[2].c_str(), sign_msg[3].c_str(),sign_msg[4].c_str());
    if(mysql_query(&conn,query)!=0)
    {  
        fprintf(stderr, "Failed to connect to databases: Error: %s\n",
        mysql_error(&conn));
    }
    else{
        puts("Sign up Success!!");
    }
    mysql_free_result(sql_result);
}

void Recv::choose_date()
{
    cout<<clnt_msg<<endl;
    vector<string> date_msg = Recv::split(clnt_msg, '/');  
    sprintf(query, "SELECT * FROM schedule WHERE Day = '%s'", date_msg[1].c_str());

    if(mysql_query(&conn,query)!=0)
    {
        fprintf(stderr, "Failed to connect to databases: Error: %s\n",
        mysql_error(&conn));           
    } 
    else
    {     
        char date_contents[SIZE_CONST::BUF_SIZE];
        sql_result = mysql_store_result(&conn);  
        while(sql_row = mysql_fetch_row(sql_result))
        {
            sprintf(date_contents, "[%s] %s/", sql_row[1], sql_row[2]);
            write(user_data->fd, date_contents, sizeof(date_contents));
            cout<<date_contents<<endl;
        }         
        write(user_data->fd,"X",sizeof("X"));   
        cout<<"X"<<endl;  
    }
    mysql_free_result(sql_result);
}

void Recv::update() 
{
    cout<<clnt_msg<<endl;
    vector<string> update_msg = Recv::split(clnt_msg, '/');     
    sprintf(query, "INSERT INTO schedule VALUES('%s', '%s', '%s')", update_msg[1].c_str(), user_data->name, update_msg[2].c_str());            
    cout<<query<<endl;

    if(mysql_query(&conn,query)!=0)
    {  
        fprintf(stderr, "Failed to connect to databases: Error: %s\n",
        mysql_error(&conn));
    }
    else{
        puts("Update Success!!");
    }         
}

vector<string> Recv::split(string str, char Delimiter) {
    istringstream iss(str);             // istringstream에 str을 담는다.
    string buffer;                      // 구분자를 기준으로 절삭된 문자열이 담겨지는 버퍼 
    vector<string> result;
 
    // istringstream은 istream을 상속받으므로 getline을 사용할 수 있다.
    while (getline(iss, buffer, Delimiter)) {
        result.push_back(buffer);               // 절삭된 문자열을 vector에 저장
    }
 
    return result;
}

void Recv::Recv_main(udata* userdata, string msg)
{
    clnt_msg = msg;
    user_data = userdata;
    if(clnt_msg.find("login") == 0)
    {
        login();
    }
    else if(clnt_msg.find("idcheck") == 0)
    {
        id_check();
    }
    else if(clnt_msg.find("signup") == 0)
    {
        sign_up();
    }    
    else if(clnt_msg.find("choose_date") == 0)
    {
        choose_date();
    }
    else if(clnt_msg.find("update") == 0)
    {        
        update();
    }
}