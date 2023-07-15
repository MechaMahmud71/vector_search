const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const similarity = require('compute-cosine-similarity');

async function loadImage(filePath) {
    return new Promise((resolve, reject) => {
        fs.readFile(filePath, (err, data) => {
            if (err) reject(err);
            resolve(data);
        });
    });
}

async function getEmbeddings(imageBuffer) {
    const model = await mobilenet.load();
    const image = tf.node.decodeImage(imageBuffer);
    const batched = image.expandDims();
    const embeddings = await model.infer(batched);
    return embeddings;
}

function displayEmbeddings(embeddings) {
    const flattenedEmbeddings = embeddings.flatten().arraySync();
    console.log("Vector Embeddings:");
    console.log(flattenedEmbeddings);
}

function cosineSimilarity(embeddings1, embeddings2) {
    const similarityScore = similarity(embeddings1, embeddings2);
    return similarityScore;
}

function euclideanDistance(embeddings1, embeddings2) {
    const distance = tf.norm(embeddings1.sub(embeddings2), 'euclidean').arraySync();
    return distance;
}


async function main() {

    const imagePath1 = './image.jpg';
    const imagePath2 = './image2.jpg';

    try {
        const imageBuffer1 = await loadImage(imagePath1);
        const embeddings1 = await getEmbeddings(imageBuffer1);
        displayEmbeddings(embeddings1);

        const imageBuffer2 = await loadImage(imagePath2);
        const embeddings2 = await getEmbeddings(imageBuffer2);
        displayEmbeddings(embeddings2);

        // const similarityScore = cosineSimilarity(embeddings1.arraySync()[0], embeddings2.arraySync()[0]);
        // console.log("Cosine Similarity:", similarityScore);

        const distance = euclideanDistance(embeddings1, embeddings2);
        console.log("Euclidean Distance:", distance);
    } catch (err) {
        console.error('Error processing the image:', err);
    }
}

main();
