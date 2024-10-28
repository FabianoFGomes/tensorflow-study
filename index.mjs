import * as tf from "@tensorflow/tfjs-node";
import XLSX from "xlsx";

const file = XLSX.readFile("./test.xlsx");
const worksheet = file.Sheets[file.SheetNames[0]];
const data = XLSX.utils.sheet_to_json(worksheet, { header: "1" });

const inputs = data.map((v) => [
  v.Bola1,
  v.Bola2,
  v.Bola3,
  v.Bola4,
  v.Bola5,
  v.Bola6,
  v.Bola7,
  v.Bola8,
  v.Bola9,
  v.Bola10,
  v.Bola11,
  v.Bola12,
  v.Bola13,
  v.Bola14,
  v.Bola15,
]);

// Função de normalização
function normalizeData(data) {
  return data.map((seq) => seq.map((num) => num / 25));
}

// Função de desnormalização com clamping
function denormalizeData(data) {
  return data.map((seq) => 
    seq.map((num) => Math.min(Math.max(Math.round(num * 25), 0), 25)) // Clamping para garantir 0-25
  );
}

// Preparar dados
const xs = tf.tensor2d(normalizeData(inputs), [inputs.length, 15]);

// Definir o modelo
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [15], units: 64, activation: "relu" }));
model.add(tf.layers.dense({ units: 128, activation: "relu" }));
model.add(tf.layers.dense({ units: 15, activation: 'sigmoid' })); // Usar sigmoid para limitar a saída entre 0 e 1

// Compilar o modelo
model.compile({
  optimizer: "adam",
  loss: 'meanSquaredError',
});

// Treinar o modelo
model
  .fit(xs, xs, {
    epochs: 50,
    batchSize: 32,
  })
  .then(() => {
    console.log("Modelo treinado com sucesso!");

    // Fazer uma predição com um input aleatório
    const input = tf.tensor2d(normalizeData([
      Array.from({ length: 15 }, () => Math.floor(Math.random() * 26))
    ]), [1, 15]);

    // Fazer a predição
    const prediction = model.predict(input);
    const predictionArray = prediction.arraySync(); // Obtém a previsão como um array de números normalizados

    // Desnormalizar e garantir a unicidade
    const result = denormalizeData(predictionArray)[0];

    // Garantir que os números sejam únicos
    const uniqueResult = [...new Set(result)].slice(0, 15);

    console.log(result, uniqueResult)
    
    // Se não tiver 15 números únicos, complete com os números que faltam
    while (uniqueResult.length < 15) {
      let num = Math.floor(Math.random() * 26);
      if (!uniqueResult.includes(num)) {
        uniqueResult.push(num);
      }
    }

    console.log('Previsão única:', uniqueResult.sort((a, b) => a - b)); // Ordena os resultados
  });
