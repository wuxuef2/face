#ifndef AAM_CONFIG_H
#define AAM_CONFIG_H
/**
 * Exception
 */
 //static char* resultPath = NULL;

//children
#define AGE_AREA 15
#define NGROUPS  8
const int AGE_GROUPS[][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3},
		{4, 4}, {5, 5}, {6, 9}, {10, 15}};  //age group



 class AgingException {
 public:
    AgingException() {
        stateCode = 0;
        init();
    }

    void init() {
/*        messageSet[0] = "Normal";
        messageSet[1] = "Cannot Read Uploaded Image";
        messageSet[2] = "Not Find Face";
        messageSet[3] = "System Busy";
        messageSet[4] = "Age Input Error";
        messageSet[5] = "Shapes != Images";
        messageSet[6] = "Age error!";
        messageSet[7] = "The image channels must be 3, and the depth must be 8!";
        messageSet[8] = "";
        messageSet[9] = "";
        messageSet[10] = "";
        messageSet[11] = "";*/
    }

    AgingException(int stateCode) {
        this->stateCode = stateCode;
        init();
    }

    void setStateCode(int stateCode) {
        this->stateCode = stateCode;
    }

    int getStateCode() {
        return stateCode;
    }

    char* getMessage() {
        return messageSet[stateCode];
    }

    char* getMessage(int code) {
        return messageSet[code];
    }
 private:
    int stateCode;
    char* messageSet[12];
 };
#endif
