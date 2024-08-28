module.exports = {
    extends: [
        "eslint:recommended",
    ],
    parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
    },
    env: {
        node: true,
        es6: true,
    },
    rules: {
        "no-unused-vars": "off",
    }
}
