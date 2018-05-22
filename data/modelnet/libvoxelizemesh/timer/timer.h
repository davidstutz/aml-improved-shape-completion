//////////////////////////////////////////////////////////////////////////////
// Timer.h
// =======
// High Resolution Timer.
// This timer is able to measure the elapsed time with 1 micro-second accuracy
// in both Windows, Linux and Unix system
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2003-01-13
// UPDATED: 2017-03-30
//
// Copyright (c) 2003 Song Ho Ahn
//////////////////////////////////////////////////////////////////////////////

#ifndef TIMER_H_DEF
#define TIMER_H_DEF

#if defined(WIN32) || defined(_WIN32)   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif
#include <stdlib.h>

class Timer
{
public:
    Timer()
    {
        #if defined(WIN32) || defined(_WIN32)
            QueryPerformanceFrequency(&frequency);
            startCount.QuadPart = 0;
            endCount.QuadPart = 0;
        #else
            startCount.tv_sec = startCount.tv_usec = 0;
            endCount.tv_sec = endCount.tv_usec = 0;
        #endif

        stopped = 0;
        startTimeInMicroSec = 0;
        endTimeInMicroSec = 0;
    };
    ~Timer()
    {

    };

    void start()
    {
        stopped = 0; // reset stop flag
        #if defined(WIN32) || defined(_WIN32)
            QueryPerformanceCounter(&startCount);
        #else
            gettimeofday(&startCount, NULL);
        #endif
    };
    void stop()
    {
        stopped = 1; // set timer stopped flag

        #if defined(WIN32) || defined(_WIN32)
            QueryPerformanceCounter(&endCount);
        #else
            gettimeofday(&endCount, NULL);
        #endif
    };
    double getElapsedTimeInMicroSec()
    {
        #if defined(WIN32) || defined(_WIN32)
            if(!stopped)
                QueryPerformanceCounter(&endCount);

            startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
            endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
        #else
            if(!stopped)
                gettimeofday(&endCount, NULL);

            startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
            endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
        #endif

            return endTimeInMicroSec - startTimeInMicroSec;
    };
    double getElapsedTimeInMilliSec()
    {
        return this->getElapsedTimeInMicroSec() * 0.001;
    };
    double getElapsedTimeInSec()
    {
        return this->getElapsedTimeInMicroSec() * 0.000001;
    };
    double getElapsedTime()
    {
        return this->getElapsedTimeInSec();
    };

private:
    double startTimeInMicroSec;
    double endTimeInMicroSec;
    int    stopped;
#if defined(WIN32) || defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER startCount;
    LARGE_INTEGER endCount;
#else
    timeval startCount;
    timeval endCount;
#endif
};

#endif // TIMER_H_DEF
