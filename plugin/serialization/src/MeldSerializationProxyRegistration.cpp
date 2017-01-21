/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "MeldForce.h"
#include "MeldForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_MELD void registerMeldSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerMeldSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerMeldSerializationProxies();
#endif

using namespace MeldPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_MELD void registerMeldSerializationProxies() {
    SerializationProxy::registerProxy(typeid(MeldForce), new MeldForceProxy());
}
