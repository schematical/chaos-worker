const config = require('config');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

class ChaosWorker{
    checkForJobs() {
        console.log("PING");
        if(this.currJobModule){
            console.log("Curr Running: ", this.currJobModule.job);
            console.log("Flushing Logs: ", this.currJobModule.flushLogs());
            //TODO: Check to see the status of the job and report
            return ;
        }
        axios.get(
            config.get("chaosnet.host") + '/jobs?state=Runnable'
        )
        .then((response)=>{
            const jobs = response.data;
            if(jobs.length > 0){
                console.log("JOBS FOUND:!", jobs);
                this.tryToStartJob(jobs[0]);
            }
        })
        .catch((err)=>{
            throw err;
        })
    }
    tryToStartJob(job){
        const jobModulePath = path.join(__dirname, 'jobs', job.type);
        if(fs.existsSync(jobModulePath)){
            console.log("Can execute: " + job.type);
            try{
                const jobClass = require(jobModulePath);
                this.currJobModule = new jobClass(job);
                this.currJobModule.run();
            }catch(err){
                throw err;
            }
        }
    }

    init(){
      // this.timeoutId = setInterval(this.checkForJobs.bind(this), config.get("check_for_jobs_interval"));
       this.checkForJobs();
    }
}
const app = new ChaosWorker();
app.init();



