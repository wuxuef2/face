################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AAM_Basic.cpp \
../AAM_CAM.cpp \
../AAM_IC.cpp \
../AAM_PAW.cpp \
../AAM_PDM.cpp \
../AAM_Shape.cpp \
../AAM_TDM.cpp \
../AAM_Util.cpp \
../AAM_VJFaceDetect.cpp \
../AgeEstimation.cpp \
../FacePredict.cpp \
../main.cpp 

OBJS += \
./AAM_Basic.o \
./AAM_CAM.o \
./AAM_IC.o \
./AAM_PAW.o \
./AAM_PDM.o \
./AAM_Shape.o \
./AAM_TDM.o \
./AAM_Util.o \
./AAM_VJFaceDetect.o \
./AgeEstimation.o \
./FacePredict.o \
./main.o 

CPP_DEPS += \
./AAM_Basic.d \
./AAM_CAM.d \
./AAM_IC.d \
./AAM_PAW.d \
./AAM_PDM.d \
./AAM_Shape.d \
./AAM_TDM.d \
./AAM_Util.d \
./AAM_VJFaceDetect.d \
./AgeEstimation.d \
./FacePredict.d \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include -I/usr/local/include/opencv -I/usr/local/include/opencv2 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


