#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "RdcForce.h"
#include "RdcForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_MELD void registerMeldSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerRdcSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerRdcSerializationProxies();
#endif

using namespace MeldPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_MELD void registerRdcSerializationProxies() {
    SerializationProxy::registerProxy(typeid(RdcForce), new RdcForceProxy());
}
