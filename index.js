const fs = require('fs');
const { promisify } = require('util');
const readFileAsync = promisify(fs.readFile);
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const similarity = require('compute-cosine-similarity');
const sharp = require('sharp');
const fetch = require('node-fetch'); // Import node-fetch for loading images from URLs
const { connect, getDb } = require("./mongodb");

async function loadImage(filePath) {
    const buffer = await readFileAsync(filePath);
    return sharp(buffer).resize(224, 224).toBuffer();
}

async function loadURL(url) {
    try {
        const fimg = await fetch(url);
        const buffer = Buffer.from(await fimg.arrayBuffer());
        return sharp(buffer).resize(224, 224).toBuffer();
    } catch (err) {
        console.error('Error fetching the image:', err);
        throw err;
    }
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

function euclideanDistance(embeddings1, embeddings2) {
    const distance = tf.norm(embeddings1.sub(embeddings2), 'euclidean').arraySync();
    return distance;
}

async function main(imageURL) {
    const imagePath1 = 'https://cdn.zdrop.com.bd/cms_files/1689604496761.jpg';
    const imageURL = 'https://cdn.zdrop.com.bd/cms_files/1689607169263.jpg';

    try {
        const imageBuffer1 = await loadURL(imageURL);
        const embeddings1 = await getEmbeddings(imageBuffer1);
        displayEmbeddings(embeddings1);

        const imageBuffer2 = await loadURL(imagePath1);
        const embeddings2 = await getEmbeddings(imageBuffer2);
        displayEmbeddings(embeddings2);

        const distance = euclideanDistance(embeddings1, embeddings2);
        console.log("Euclidean Distance:", distance);
    } catch (err) {
        console.error('Error processing the image:', err);
    }
}



// main();


