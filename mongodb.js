const { MongoClient } = require('mongodb');

let db;

const connect = async () => {
    const client = new MongoClient(process.env.MONGODB);
    await client.connect();
    db = client.db("product_db")
}
const getDb = async () => {
    if (!db) await connect();
    return db;
}

module.exports = { connect, getDb }