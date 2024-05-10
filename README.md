# Game_Secret_AI
Secret file transfert through AI encoding and decoding

### Game Name: SecreatAI

#### Objective:
The 
goal of SecreatAI is to encode and decode files using an autoencoder, while preventing an opponent (either a computer or another player) from intercepting and decoding the file.

#### Necessary Equipment:
- A device capable of running the provided Python file.

#### Rules:
1. *Preparation*:
   - Each player must train their own autoencoder on a random binary corpus. This corpus is customized for each player to create unique encoding models.

2. *Encoding the File*:
   - A player selects a file and uses it as a binary input for their autoencoder, which produces an encoded file.
   - The same player tests decoding the file to ensure it can be reconstructed with 100% accuracy before sending it.

3. *Transmitting the Encoded File*:
   - The encoded file is sent to the other player through an agreed-upon communication channel.

4. *Attempt to Decode*:
   - The opponent attempts to decode the file using their own autoencoder. If the corpus on which the opponent's autoencoder was trained is sufficiently similar, they might succeed in decoding the file. Otherwise, the message remains indecipherable. All other methods are welcome.

5. *Revelation and Scoring*:
   - The original player reveals the original file. If the opponent has correctly decoded the file, they earn points. Otherwise, the player who encoded the file wins points.

6. *Rotation*:
   - Roles are reversed for the next round.

#### Technical Development:
- *Autoencoder*: Use a simple autoencoder architecture with dense layers and a sigmoid activation function for encoding and decoding. Libraries such as TensorFlow or PyTorch can be used to implement the autoencoder. Alternatively, you can analyze, modify, and use the code attached.
- *User Interface*: Develop a simple interface where players can encode their files, view the encoded files, and attempt to decode them.
- *Database*: Use a distinct corpus for each player for training the autoencoder, ensuring unique encoding patterns, which is the case each time the program is launched for the first time.

This framework provides a structured, interactive way to explore machine learning concepts and cryptography in a game setting, suitable for those interested in these fields.
