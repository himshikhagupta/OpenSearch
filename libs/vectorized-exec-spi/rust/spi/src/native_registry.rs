// WHAT THIS DOES:
//   Allows plugins to register their Java native bridge classes so that
//   JNI native methods declared in child classloaders can be bound to
//   Rust function pointers in a single shared .so loaded by the parent classloader.
//
// HOW PLUGINS USE IT:
//   In each plugin's Rust lib.rs, add one line:
//
//     vectorized_exec_spi::register_native_bridge!("org.opensearch.datafusion.jni.NativeBridge");
//
//   That's it. When Java calls NativeLibraryLoader.registerClass(NativeBridge.class),
//   this module automatically:
//     1. Reflects over the Java class to find all `native` methods
//     2. Builds JNI type signatures from the method's parameter/return types
//     3. Looks up the corresponding Rust function via dlsym
//     4. Calls JNI RegisterNatives to bind them
//
// HOW AUTO-DISCOVERY WORKS (step by step):
//
//   Given this Java class:
//     public class NativeBridge {
//         public static native String getVersionInfo();
//         public static native void initTokioRuntimeManager(int cpuThreads);
//     }
//
//   And this Rust code:
//     #[no_mangle]
//     pub extern "system" fn Java_org_opensearch_datafusion_jni_NativeBridge_getVersionInfo(...) { ... }
//
//   The auto-discovery does:
//
//   a) Java Reflection (via JNI calls from Rust):
//      - class.getDeclaredMethods() → [getVersionInfo, initTokioRuntimeManager, ...]
//      - For each method, check method.getModifiers() & 0x0100 (NATIVE flag)
//      - method.getName() → "getVersionInfo"
//      - method.getParameterTypes() → [] (no params)
//      - method.getReturnType() → java.lang.String
//
//   b) JNI Signature Construction:
//      - Map Java types to JNI descriptors:
//          void → "V"       int → "I"        long → "J"       boolean → "Z"
//          String → "Ljava/lang/String;"      byte[] → "[B"
//          ActionListener → "Lorg/opensearch/core/action/ActionListener;"
//      - Combine: "()" + "Ljava/lang/String;" → "()Ljava/lang/String;"
//      - For initTokioRuntimeManager(int): "(I)V"
//
//   c) Symbol Lookup (dlsym):
//      - Build symbol name: "Java_" + "org_opensearch_datafusion_jni_NativeBridge" + "_" + "getVersionInfo"
//        (dots in package name replaced with underscores — standard JNI naming convention)
//      - dlsym(RTLD_DEFAULT, "Java_org_opensearch_datafusion_jni_NativeBridge_getVersionInfo")
//        → returns function pointer 0x00000000000c43ac
//      - RTLD_DEFAULT searches all loaded shared libraries in the process, so it finds
//        the symbol in our .so regardless of which classloader loaded it
//
//   d) JNI RegisterNatives:
//      - Calls env.RegisterNatives(NativeBridge.class, [
//          { name: "getVersionInfo", sig: "()Ljava/lang/String;", fn_ptr: 0x...c43ac },
//          { name: "initTokioRuntimeManager", sig: "(I)V", fn_ptr: 0x...c2a40 },
//        ])
//      - JVM patches NativeBridge's method table with direct function pointers
//      - All future calls to NativeBridge.getVersionInfo() jump directly to 0x...c43ac
//        — no classloader lookup, no symbol resolution, just a direct function call
//
// PERFORMANCE:
//   - This entire process runs ONCE per plugin at JVM startup (in the static {} block)
//   - After RegisterNatives, native calls are direct pointer jumps — zero overhead
//   - The reflection + dlsym cost is ~microseconds, completely negligible at startup
//
// ADDING A NEW NATIVE METHOD (developer workflow):
//   1. Add `public static native Foo bar(...)` in the Java bridge class
//   2. Add `#[no_mangle] pub extern "system" fn Java_..._bar(...)` in the plugin's Rust code
//   3. Done. No registration list to update, no signatures to write manually.
//
// ADDING A NEW PLUGIN:
//   1. Add `register_native_bridge!("com.example.NewBridge")` in the new Rust rlib
//   2. Add `NativeLibraryLoader.registerClass(NewBridge.class)` in Java's static block
//   3. Add `extern crate new_plugin;` in jni-entry/Cargo.toml + lib.rs
//   4. Done.
//
// ═══════════════════════════════════════════════════════════════════════════════

use jni::JNIEnv;
use jni::objects::JClass;

/// A registrar entry — just the fully qualified Java class name.
/// Methods are discovered automatically at runtime via reflection + dlsym.
///
/// Populated at link time by the `register_native_bridge!` macro in each plugin rlib.
pub struct NativeBridgeRegistrar {
    pub class_name: &'static str,
}

/// Distributed slice — a compile-time array that each plugin rlib appends to.
/// When the final cdylib is linked, this slice contains one entry per plugin.
/// At runtime, jni-entry iterates this to verify a class was registered before
/// attempting to bind its native methods.
///
/// This uses the `linkme` crate which works by placing each entry in a special
/// linker section, then collecting them into a contiguous slice at link time.
#[linkme::distributed_slice]
pub static NATIVE_REGISTRARS: [NativeBridgeRegistrar];

/// Macro for plugins to register their native bridge class.
///
/// This adds an entry to NATIVE_REGISTRARS at link time. No method list needed —
/// methods are discovered automatically when registerClass() is called from Java.
///
/// # Example
/// ```ignore
/// // In engine-datafusion/jni/src/lib.rs — just this one line:
/// vectorized_exec_spi::register_native_bridge!("org.opensearch.datafusion.jni.NativeBridge");
/// ```
#[macro_export]
macro_rules! register_native_bridge {
    ($class_name:expr) => {
        #[linkme::distributed_slice($crate::native_registry::NATIVE_REGISTRARS)]
        static _REGISTRAR: $crate::native_registry::NativeBridgeRegistrar =
            $crate::native_registry::NativeBridgeRegistrar {
                class_name: $class_name,
            };
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Auto-discovery: Reflection + dlsym + RegisterNatives
// ═══════════════════════════════════════════════════════════════════════════════

/// Reflects over a Java class's native methods and binds them to Rust function pointers.
///
/// For each `native` method in the class:
///   1. Gets the method name via `java.lang.reflect.Method.getName()`
///   2. Builds the JNI type signature from `getParameterTypes()` and `getReturnType()`
///   3. Constructs the JNI symbol name: `Java_<package_class>_<method>`
///   4. Looks up the symbol in the current process via `dlsym(RTLD_DEFAULT, ...)`
///   5. Collects all {name, signature, function_pointer} tuples
///   6. Calls `env.RegisterNatives(class, methods)` to bind them all at once
///
/// Called once per plugin at startup. After this, native calls are direct jumps.
pub fn register_natives_for_class(env: &mut JNIEnv, target_class: &JClass) -> Result<(), String> {
    let class_name = get_class_name(env, target_class)?;

    // ── Step 1: Get all declared methods via Java reflection ──
    // Equivalent to: Method[] methods = NativeBridge.class.getDeclaredMethods();
    let methods_array = env.call_method(target_class, "getDeclaredMethods", "()[Ljava/lang/reflect/Method;", &[])
        .map_err(|e| format!("getDeclaredMethods: {}", e))?.l().map_err(|e| e.to_string())?;
    let arr: jni::objects::JObjectArray = methods_array.into();
    let len = env.get_array_length(&arr).map_err(|e| e.to_string())?;

    // java.lang.reflect.Modifier.NATIVE = 0x0100 (256)
    const NATIVE: i32 = 0x0100;
    let mut native_methods = Vec::new();

    for i in 0..len {
        let method = env.get_object_array_element(&arr, i).map_err(|e| e.to_string())?;

        // ── Step 2: Check if this method is `native` ──
        // Equivalent to: if ((method.getModifiers() & Modifier.NATIVE) == 0) continue;
        let mods = env.call_method(&method, "getModifiers", "()I", &[])
            .map_err(|e| e.to_string())?.i().map_err(|e| e.to_string())?;
        if mods & NATIVE == 0 { continue; }

        // ── Step 3: Get method name ──
        // Equivalent to: String name = method.getName(); // e.g. "getVersionInfo"
        let name = get_string(env, &method, "getName")?;

        // ── Step 4: Build JNI type signature from parameter and return types ──
        // e.g. getVersionInfo() → "()Ljava/lang/String;"
        // e.g. executeQueryPhaseAsync(long, String, byte[], boolean, int, long, ActionListener)
        //   → "(JLjava/lang/String;[BZIJLorg/opensearch/core/action/ActionListener;)V"
        let sig = build_jni_signature(env, &method)?;

        // ── Step 5: Build JNI symbol name and look it up via dlsym ──
        // Convention: Java_<package_class>_<method> with dots replaced by underscores
        // e.g. "Java_org_opensearch_datafusion_jni_NativeBridge_getVersionInfo"
        let symbol = format!("Java_{}_{}", class_name.replace('.', "_"), name);
        let symbol_c = std::ffi::CString::new(symbol.clone()).map_err(|e| e.to_string())?;

        // dlsym(RTLD_DEFAULT, ...) searches ALL loaded shared libraries in the process.
        // Since our .so is loaded (by the parent classloader), the symbol is found here
        // regardless of classloader boundaries — dlsym doesn't know about Java classloaders.
        let fn_ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol_c.as_ptr()) };
        if fn_ptr.is_null() {
            return Err(format!("Symbol not found: {} — is the #[no_mangle] function defined in Rust?", symbol));
        }

        native_methods.push(jni::NativeMethod {
            name: name.into(),
            sig: sig.into(),
            fn_ptr,
        });
    }

    if native_methods.is_empty() {
        return Err(format!("No native methods in {}", class_name));
    }

    // ── Step 6: Call JNI RegisterNatives ──
    // This tells the JVM: "for this Class object (from child classloader), when method X
    // is called, jump directly to function pointer Y." After this call, the JVM has
    // patched the class's method table with direct pointers — no more symbol lookup needed.
    env.register_native_methods(target_class, &native_methods)
        .map_err(|e| {
            if let Ok(true) = env.exception_check() {
                env.exception_describe().ok();
                env.exception_clear().ok();
            }
            format!("RegisterNatives failed for {}: {}", class_name, e)
        })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper functions for JNI reflection
// ═══════════════════════════════════════════════════════════════════════════════

/// Get the fully qualified class name from a Class<?> object.
/// e.g. "org.opensearch.datafusion.jni.NativeBridge"
pub fn get_class_name(env: &mut JNIEnv, cls: &JClass) -> Result<String, String> {
    get_string_from_obj(env, cls.as_ref(), "getName")
}

fn get_string(env: &mut JNIEnv, obj: &jni::objects::JObject, method: &str) -> Result<String, String> {
    get_string_from_obj(env, obj, method)
}

/// Call a no-arg method that returns String on a Java object, convert to Rust String.
fn get_string_from_obj(env: &mut JNIEnv, obj: &jni::objects::JObject, method: &str) -> Result<String, String> {
    let jstr = env.call_method(obj, method, "()Ljava/lang/String;", &[])
        .map_err(|e| e.to_string())?.l().map_err(|e| e.to_string())?;
    let s = env.get_string((&jstr).into()).map_err(|e| e.to_string())?;
    Ok(s.to_string_lossy().to_string())
}

/// Build a JNI method signature string from a java.lang.reflect.Method.
///
/// Uses reflection to get parameter types and return type, then converts each
/// to its JNI descriptor:
///   - Primitives: int→"I", long→"J", boolean→"Z", void→"V", etc.
///   - Objects: String→"Ljava/lang/String;", ActionListener→"Lorg/opensearch/core/action/ActionListener;"
///   - Arrays: byte[]→"[B", String[]→"[Ljava/lang/String;", long[]→"[J"
///
/// Example: executeQueryPhaseAsync(long, String, byte[], boolean, int, long, ActionListener) → void
///   Parameters: J, Ljava/lang/String;, [B, Z, I, J, Lorg/opensearch/core/action/ActionListener;
///   Return: V
///   Result: "(JLjava/lang/String;[BZIJLorg/opensearch/core/action/ActionListener;)V"
fn build_jni_signature(env: &mut JNIEnv, method: &jni::objects::JObject) -> Result<String, String> {
    // Get parameter types: Class<?>[] params = method.getParameterTypes();
    let params = env.call_method(method, "getParameterTypes", "()[Ljava/lang/Class;", &[])
        .map_err(|e| e.to_string())?.l().map_err(|e| e.to_string())?;
    let arr: jni::objects::JObjectArray = params.into();
    let len = env.get_array_length(&arr).map_err(|e| e.to_string())?;

    let mut sig = String::from("(");
    for i in 0..len {
        let cls = env.get_object_array_element(&arr, i).map_err(|e| e.to_string())?;
        sig.push_str(&class_to_descriptor(env, &cls)?);
    }
    sig.push(')');

    // Get return type: Class<?> ret = method.getReturnType();
    let ret = env.call_method(method, "getReturnType", "()Ljava/lang/Class;", &[])
        .map_err(|e| e.to_string())?.l().map_err(|e| e.to_string())?;
    sig.push_str(&class_to_descriptor(env, &ret)?);
    Ok(sig)
}

/// Convert a java.lang.Class to its JNI type descriptor string.
///
/// Uses class.getName() which returns:
///   - Primitives: "int", "long", "boolean", "void", etc.
///   - Objects: "java.lang.String", "org.opensearch.core.action.ActionListener"
///   - Arrays: "[B" (byte[]), "[Ljava.lang.String;" (String[])
///
/// We convert to JNI descriptors:
///   - "int" → "I"
///   - "java.lang.String" → "Ljava/lang/String;"
///   - "[Ljava.lang.String;" → "[Ljava/lang/String;" (replace dots with slashes)
fn class_to_descriptor(env: &mut JNIEnv, cls: &jni::objects::JObject) -> Result<String, String> {
    let name = get_string_from_obj(env, cls, "getName")?;
    Ok(match name.as_str() {
        "void" => "V", "boolean" => "Z", "byte" => "B", "char" => "C",
        "short" => "S", "int" => "I", "long" => "J", "float" => "F", "double" => "D",
        n if n.starts_with('[') => return Ok(n.replace('.', "/")),
        n => return Ok(format!("L{};", n.replace('.', "/"))),
    }.to_string())
}
