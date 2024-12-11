#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>

// Defina o SSID e a senha para a rede Wi-Fi do ESP32 (Access Point)
const char* ssid = "KINEARM";
const char* password = "12345678"; // A senha deve ter pelo menos 8 caracteres

WiFiServer server(5005); // Porta para comunicação TCP

void setup() {
  Serial.begin(115200);

  // Configura o ESP32 como Access Point
  Serial.println("Configurando o Access Point...");
  WiFi.softAP(ssid, password);

  IPAddress IP = WiFi.softAPIP();
  Serial.print("Endereço IP do AP: ");
  Serial.println(IP);

  // Inicia o servidor TCP
  server.begin();
  Serial.println("Servidor TCP iniciado.");
}

void loop() {
  WiFiClient client = server.available(); // Verifica se há clientes conectados

  if (client) {
    Serial.println("Cliente conectado.");
    while (client.connected()) {
      // Leia os dados dos sensores ou variáveis correspondentes
      float ang_pun = map(random(0, 1000), 0, 1000, 0, 180); // Substitua pela leitura real de ang_pun
      float ang_cot = map(random(0, 1000), 0, 1000, 0, 180); // Substitua pela leitura real de ang_cot
      float ace_mao = map(random(0, 1000), 0, 1000, -1, 1); // Substitua pela leitura real de ace_mao
      float ace_ant = map(random(0, 1000), 0, 1000, -1, 1); // Substitua pela leitura real de ace_ant

      // Formate os dados em uma string (por exemplo, CSV)
      String data = String(ang_pun) + "," + String(ang_cot) + "," + String(ace_mao) + "," + String(ace_ant) + "\n";

      // Envie os dados para o cliente
      client.print(data);

      delay(50); // 50ms - 20Hz
    }
    client.stop();
    Serial.println("Cliente desconectado.");
  }
}
