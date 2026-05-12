/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Proc macro for FFM bridge functions.
//!
//! Wraps an `extern "C"` function body with `catch_unwind`. The body must
//! return `Result<i64, String>`. On success the `i64` is returned directly.
//! On `Err` or panic, the error message is heap-allocated and returned as a
//! negative pointer (negated `Box::into_raw` address).
//!
//! Java checks: if result < 0, call `native_error_message(-result)` to get
//! the message, then `native_error_free(-result)` to free it.
//!
//! # Usage
//!
//! ```ignore
//! #[ffm_safe]
//! #[no_mangle]
//! pub unsafe extern "C" fn my_func(arg: i64) -> i64 {
//!     do_work(arg).map_err(|e| e.to_string())
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};
use syn::parse::{Parse, ParseStream};

/// Optional attribute: `#[ffm_safe(plugin = crate::PLUGIN_ID)]`
/// The path must point to a `OnceLock<PluginHandle>` static.
struct FfmSafeAttr {
    plugin_path: Option<syn::Path>,
}

impl Parse for FfmSafeAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self { plugin_path: None });
        }
        let ident: syn::Ident = input.parse()?;
        if ident != "plugin" {
            return Err(syn::Error::new(ident.span(), "expected `plugin`"));
        }
        let _: syn::Token![=] = input.parse()?;
        let path: syn::Path = input.parse()?;
        Ok(Self { plugin_path: Some(path) })
    }
}

#[proc_macro_attribute]
pub fn ffm_safe(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as FfmSafeAttr);
    let input = parse_macro_input!(item as ItemFn);
    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let body = &input.block;

    let fn_name = input.sig.ident.to_string();

    let bind_call = args.plugin_path.map(|path| {
        quote! {
            native_bridge_common::allocator::bind_thread(
                #path.get().expect("plugin not registered: ensure init is called first")
            );
        }
    });

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            #bind_call
            native_bridge_common::error::ffm_wrap(
                #fn_name,
                ::std::panic::AssertUnwindSafe(
                    || -> ::std::result::Result<i64, ::std::string::String> #body
                ),
            )
        }
    };

    expanded.into()
}
