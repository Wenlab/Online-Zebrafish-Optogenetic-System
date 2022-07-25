#include"TCPServer.h"

#include <string>                                                                        
#include <iostream>
#include <cstring>
#include <Ws2tcpip.h>

#pragma comment(lib,"ws2_32.lib")

using namespace std;

void TCPServer::initialize()
{
	int result = 0;
	result = WSAStartup(MAKEWORD(2, 2), &data_);//inital
	if (result != 0)
	{
		std::cout << "WSAStartup() init error " << GetLastError() << std::endl;
		//system("pause");
		return;
	}

	server_ = socket(AF_INET, SOCK_STREAM, 0);

	SOCKADDR_IN addrSrv;
	addrSrv.sin_family = AF_INET;
	addrSrv.sin_port = htons(11000);
	addrSrv.sin_addr.S_un.S_addr = htonl(INADDR_ANY);//ip port 

	result = bind(server_, (LPSOCKADDR)&addrSrv, sizeof(SOCKADDR_IN));

	if (result != 0)
	{	
		std::cout << "bind() error" << result;
		//system("pause");
		return;
	}
	result = listen(server_, 10);
	if (result != 0)
	{
		std::cout << "listen error" << result;
		//system("pause");
		return;
	}

	SOCKADDR_IN addrClient;
	int len = sizeof(SOCKADDR);

	std::cout << "wait new connect......" << std::endl;
	socketConn2 = accept(server_, (SOCKADDR*)&addrClient, &len);

	int Timeout = 4;
	setsockopt(socketConn2, SOL_SOCKET, SO_RCVTIMEO, (char*)&Timeout, sizeof(int));

	if (socketConn2 == SOCKET_ERROR)
	{
		std::cout << " accept error" << WSAGetLastError();
		return;
	}
	std::cout << "connect successful" << std::endl;

	return;
}



void TCPServer::receive()
{
	char recvBuff[1024];
	memset(recvBuff, 0, sizeof(recvBuff));
	recv(socketConn2, recvBuff, sizeof(recvBuff), 0);

	//std::cout << "client say:" << recvBuff << std::endl;

	data = string2int(recvBuff);   //这一句会导致程序卡住？？？
	//cout << data << endl;

	Sleep(1);
}

void TCPServer::close()
{
	closesocket(server_);
	WSACleanup();
}

int string2int(char *arg)
{
	char c;
	int i, j, tmp = 0, tmpc = 0;
	for (i = 0; i < strlen(arg); i++) {
		c = (*(arg + i));
		tmp = tmp * 10 + (int)c - 48;
	}
	return tmp;
}