/*
 * Example code from: https://github.com/bozkurthan/Simple-TCP-Server-Client-CPP-Example
 *
 * Minor cleanups introduced for clarity.
 */
#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/time.h>

//Server side
int main(int argc, char *argv[])
{
  //for the server, we only need to specify a port number
  if (argc != 2) {
    std::cerr << "Usage: port" << std::endl;
    exit(0);
  }
  //grab the port number
  int port = atoi(argv[1]);
  //buffer to send and receive messages with
  char msg[1500];

  //setup a socket and connection tools
  sockaddr_in servAddr;
  bzero((char *) &servAddr, sizeof(servAddr));
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  servAddr.sin_port = htons(port);

  //open stream oriented socket with internet address
  //also keep track of the socket descriptor
  int serverSd = socket(AF_INET, SOCK_STREAM, 0);
  if (serverSd < 0) {
    std::cerr << "Error establishing the server socket" << std::endl;
    exit(0);
  }
  //bind the socket to its local address
  int bindStatus = bind(serverSd, (struct sockaddr *) &servAddr,
                        sizeof(servAddr));
  if (bindStatus < 0) {
    std::cerr << "Error binding socket to local address" << std::endl;
    exit(0);
  }
  std::cout << "Waiting for a client to connect..." << std::endl;
  //listen for up to 5 requests at a time
  listen(serverSd, 5);
  std::cout << "listening..." << std::endl;
  //receive a request from client using accept
  //we need a new address to connect with the client
  sockaddr_in newSockAddr;
  socklen_t newSockAddrSize = sizeof(newSockAddr);
  //accept, create a new socket descriptor to 
  //handle the new connection with client
  int newSd = accept(serverSd, (sockaddr *) &newSockAddr, &newSockAddrSize);
  if (newSd < 0) {
    std::cerr << "Error accepting request from client!" << std::endl;
    exit(1);
  }
  std::cout << "Connected with client!" << std::endl;
  //lets keep track of the session time
  struct timeval start1, end1;
  gettimeofday(&start1, nullptr);
  //also keep track of the amount of data sent as well
  int bytesRead, bytesWritten = 0;
  while (1) {
    //receive a message from the client (listen)
    std::cout << "Awaiting client response..." << std::endl;
    memset(&msg, 0, sizeof(msg));//clear the buffer
    bytesRead += recv(newSd, (char *) &msg, sizeof(msg), 0);
    if (!strcmp(msg, "exit")) {
      std::cout << "Client has quit the session" << std::endl;
      break;
    }
    std::cout << "Client: " << msg << std::endl;
    std::cout << ">";
    std::string data;
    getline(std::cin, data);
    memset(&msg, 0, sizeof(msg)); //clear the buffer
    strcpy(msg, data.c_str());
    if (data == "exit") {
      //send to the client that server has closed the connection
      send(newSd, (char *) &msg, strlen(msg), 0);
      break;
    }
    //send the message to client
    bytesWritten += send(newSd, (char *) &msg, strlen(msg), 0);
  }
  //we need to close the socket descriptors after we're all done
  gettimeofday(&end1, nullptr);
  close(newSd);
  close(serverSd);
  std::cout << "********Session********" << std::endl;
  std::cout << "Bytes written: " << bytesWritten << " Bytes read: " << bytesRead << std::endl;
  std::cout << "Elapsed time: " << (end1.tv_sec - start1.tv_sec)
       << " secs" << std::endl;
  std::cout << "Connection closed..." << std::endl;
  return 0;
}
