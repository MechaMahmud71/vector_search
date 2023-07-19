const express = require("express")
const app = express();
const fs = require('fs');
const { promisify } = require('util');
const readFileAsync = promisify(fs.readFile);
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const similarity = require('compute-cosine-similarity');
const sharp = require('sharp');
const fetch = require('node-fetch'); // Import node-fetch for loading images from URLs
const { connect, getDb } = require("./mongodb");
const dotenv = require('dotenv');

dotenv.config({ path: "./.env" });

app.use(express.json());


async function getEmbeddings(imageBuffer) {
    const model = await mobilenet.load();
    const image = tf.node.decodeImage(imageBuffer);
    const batched = image.expandDims();
    const embeddings = await model.infer(batched);
    return embeddings;
}

// function displayEmbeddings(embeddings) {
//     const flattenedEmbeddings = embeddings.flatten().arraySync();
//     console.log("Vector Embeddings:");
//     console.log(flattenedEmbeddings);
// }

function displayEmbeddings(embeddings) {
    const flattenedEmbeddings = embeddings.flatten().arraySync();
    return flattenedEmbeddings;
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



app.post("/update-docs", async (req, res, next) => {
    try {
        const productDb = await getDb();
        const products = await productDb.collection("products").find().toArray();

        for (let i = 0; i < products.length; i++) {
            let imageBuffer1 = await loadURL(products[i].image);
            let embeddings1 = await getEmbeddings(imageBuffer1);
            let emb1 = displayEmbeddings(embeddings1);

            // products[i].embeddings = emb1;

            let product = await productDb.collection('products').updateOne({
                _id: products[i]._id
            }, {
                $set: {
                    embeddings: emb1
                }
            }, { new: true })

            console.log(product)

        }
        // console.log(products)
        res.json({
            success: true,
            data: products
        })
    } catch (error) {
        console.log(error);
        throw new Error(error.message)
    }

})


app.get("/search-product", async (req, res, next) => {
    try {
        const productDb = await getDb();
        let imageBuffer1 = await loadURL("https://cdn.zdrop.com.bd/cms_files/1689782082684.jpg");
        let embeddings1 = await getEmbeddings(imageBuffer1);
        let emb1 = displayEmbeddings(embeddings1);
        const documents = await productDb.collection("products").aggregate([
            {
                "$search": {
                    "index": "default",
                    "knnBeta": {
                        "vector": emb1,
                        "path": "embeddings",
                        "k": 3
                    }
                }
            }
        ]).toArray();

        return res.json({
            success: true,
            data: documents
        })
    } catch (error) {
        return res.json({
            success: false,
            message: error.message
        })
    }
})


app.listen(3000, async () => {
    await connect();
    console.log("App is running at port 3000")
    console.log("http://localhost:3000")
})