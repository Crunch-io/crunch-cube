# See http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
try:  # pragma: no cover
    __import__("pkg_resources").declare_namespace(__name__)
except ImportError:  # pragma: no cover
    __import__("pkgutil").extend_path(__path__, __name__)  # type: ignore
