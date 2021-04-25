const axios = require('axios');
const config = require('config');
const tf = require("@tensorflow/tfjs-node");
const AdmZip = require('adm-zip');
const path = require('path');
const fs = require('fs');
class RunnableJobBase{

    constructor(job) {
        this._job = job;
        this._jobMeta = this._job.jobMeta;
        this._logBuffer = '';
        this._stages = [];
    }
    get job(){
        return this._job;
    }
    get jobMeta(){
        return this._jobMeta;
    }
    log(){
        this._logBuffer += Array.prototype.slice.call(arguments).join("       ") + "\n";
    }
    flushLogs(){
        return this._logBuffer;
        this._logBuffer = '';
    }


    async downloadFile(fileUrl, outputLocationPath) {

        return axios({
            method: 'get',
            url: fileUrl,
            headers: {
                Authorization: this.getAuthorization()
            }
        }).then(response => {
            return axios({
                method: 'get',
                url: response.data.url,
            })
        })
        .then(response => {
            if(outputLocationPath) {
                fs.writeFileSync(outputLocationPath, response.data);
            }
            return response.data;
        });
    }
    async saveFile(inputLocationPath, fileUrl) {
        let fileBody = fs.readFileSync(inputLocationPath).toString();
        return axios({
            method: 'post',
            url: fileUrl,
            headers: {
                Authorization: this.getAuthorization()
            }
        }).then(response => {
            return axios.put(response.data.url, fileBody, {
                /*  headers: {
                      'Content-Type': 'application/json'
                  }*/
            });
        });
    }
    getAuthorization(){
        let creds = config.get('chaosnet.username') + ':' + config.get('chaosnet.secretKey');
        let buffer = new Buffer(creds);
        return 'Basic ' + buffer.toString('base64');
    }
    saveModel(model, fileUrl) {
        const modelSaveURL = 'file://./tmp/' + this.job._id + '-v3';
        const modelSavePath = modelSaveURL.replace('file://', '');
        const dirName = path.dirname(modelSavePath);
        if (!fs.existsSync(dirName)) {
            fs.mkdirSync(dirName);
        }
        let zipBuffer = null;
        return model.save(modelSaveURL)
            .then(() => {

                this.log(`Trained model is saved to ${modelSaveURL}`);
                this.log(
                    `\nNext, run the following command to test the model in the browser:`);
                this.log(`\n  yarn watch`);
                var zip = new AdmZip();

                // add file directly
                var content = "inner content of the file";
                zip.addFile('model.json', fs.readFileSync(modelSavePath + '/model.json'), "");
                zip.addFile('weights.bin', fs.readFileSync(modelSavePath + '/weights.bin'), "");

                // get everything as a buffer
                zipBuffer = zip.toBuffer();


                const auth = this.getAuthorization();
                return axios({
                    method: 'post',
                    url: fileUrl,
                    headers: {
                        Authorization: auth
                    }
                })
            })
            .then(response => {
                return axios({
                    method: 'put',
                    url: response.data.url,
                    data: zipBuffer,
                    'maxContentLength': Infinity,
                    'maxBodyLength': Infinity
                })

            })
            .then((response) => {
                this.log("I think the upload worked? ", response)
            })
            .catch((err) => {
                console.error(err);
            })
    }
    updateStatus(status) {
        let url = config.get("chaosnet.host") + '/jobs/' + this.job._id + '/status';
        return axios.post(
            url,
            {
                status: status
            },
            {
                headers: {
                    Authorization: this.getAuthorization()
                }
            }
        )
        .catch((err) => {
            console.error(err);
        })
    }
}
module.exports = RunnableJobBase;
