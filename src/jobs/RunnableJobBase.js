const axios = require('axios');
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
    async updateStatus(status) {

        return axios(
            {
                method: 'post',
                url: '/jobs/' + this.job._id + '/status',
            },
            status
        );
    }
}
module.exports = RunnableJobBase;
